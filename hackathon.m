load('trainedYOLOv4.mat','detector');

%% Select PCD files
[files, filePath] = uigetfile('*.pcd', 'Select one or more PCD files', 'MultiSelect', 'on');
if isequal(files,0)
    error('No PCD file selected.');
end
if ischar(files)
    files = {files}; % ensure cell array
end
pcdFiles = fullfile(filePath, files);

pcds = fileDatastore(pcdFiles,'ReadFcn',@(x) pcread(x));

%% Define BEV parameters
% xMin = 101070.0; xMax = 101170.0;
% yMin = 85350.0;   yMax = 85450.0;
xMin = 101100.0; xMax = 101150.0;
yMin = 85400.0;   yMax = 85450.0;
zMin = 140.0;  zMax = 220.0;

bevHeight = 608;
bevWidth  = 608;

gridW = (yMax - yMin)/bevWidth;
gridH = (xMax - xMin)/bevHeight;

gridParams = {{xMin,xMax,yMin,yMax,zMin,zMax}, {bevWidth,bevHeight}, {gridW,gridH}};

classNames = {'Car','Truck','Pedestrain'};

%% Run detection on each PCD file
reset(pcds)
while hasdata(pcds)
    ptCld = read(pcds);

    % Convert point cloud to BEV image & recize for detector
    [bevImage, ptCldOut] = preprocess(ptCld, gridParams, xMin, yMin);
    % [bevImage, ptCldOut] = preprocess(ptCld, gridParams);
    I = im2single(imresize(bevImage, [608 608]));

    % Run detector
    [bboxes, scores, labels] = detect(detector, I);

    % Show detections on BEV image
    helperDisplayBoxes(I, bboxes, labels);
    title('Detections on BEV image');
    

    % Map detections back to 3D point cloud using yolo boxes
    [ptCldDet, bboxCuboid] = transferbboxToPointCloud(bboxes, gridParams, ptCld);
    helperDisplayBoxes(ptCldDet,bboxCuboid,labels);

    % % debugging
    % disp('GT bounding boxes format:')
    % whos bboxes
    % if ~isempty(bboxes)
    %     for i = 1:3;
    %         disp(bboxes(i,:)) % [x, y, w, h, yaw]
    %     end
    % end

end


%% --- Supporting functions ---
function [ptCldOut,bboxCuboid] = transferbboxToPointCloud(bboxes,gridParams,ptCld)
% Transfer labels from images to point cloud.
    
    % Crop the point cloud.
    pcRange = [gridParams{1,1}{1} gridParams{1,1}{2} gridParams{1,1}{3} ...
               gridParams{1,1}{4} gridParams{1,1}{5} gridParams{1,1}{6}]; 

    indices = findPointsInROI(ptCld,pcRange);
    ptCldOut = select(ptCld,indices);

    % Assume height of objects to be a constant based on input data.
    objectHeight = 2.2;

    % Grid params
    xMin  = gridParams{1}{1};
    yMin  = gridParams{1}{3};
    gridW = gridParams{3}{1}; % along Y
    gridH = gridParams{3}{2}; % along X
    zMin  = gridParams{1}{5};
    zMax  = gridParams{1}{6};
    
    % Calculate the height of the ground plane.
    groundPtsIdx = segmentGroundSMRF(ptCldOut,3,'MaxWindowRadius',5,'ElevationThreshold',0.4,'ElevationScale',0.25);
    loc = ptCldOut.Location;
    groundHeight = mean(loc(groundPtsIdx,3));

    if isempty(bboxes)
        bboxCuboid = zeros(0,9);
        return;
    end
    
    xCenter = xMin + (bboxes(:,2) + bboxes(:,4)/2) * gridH;
    yCenter = yMin + (bboxes(:,1) + bboxes(:,3)/2) * gridW;
    xMinCuboid = xMin + bboxes(:,2) * gridH;
    yMinCuboid = yMin + bboxes(:,1) * gridW;

    bboxCuboid = zeros(size(bboxes,1),9);
    bboxCuboid(:,1) = xMinCuboid;
    bboxCuboid(:,2) = yMinCuboid;
    bboxCuboid(:,4) = bboxes(:,4) * gridH;
    bboxCuboid(:,5) = bboxes(:,3) * gridW;
    bboxCuboid(:,9) = -bboxes(:,5);
    
    bboxCuboid(:,6) = (objectHeight) * ones(size(xCenter));
    bboxCuboid(:,3) = (groundHeight + objectHeight/2) * ones(size(xCenter));

    % disp(groundHeight);
    % disp(zMin);
    % disp(bboxCuboid(:,3));

end

function helperDisplayBoxes(obj,bboxes,labels)
% Display the boxes over the image and point cloud.

    figure
    if ~isa(obj,'pointCloud')
        imshow(obj)
        shape = 'rectangle';
    else
        pcshow(obj.Location);
        shape = 'cuboid';
    end
    hold on;
    showShape(shape,bboxes(labels=='Car',:),...
                  'Color','green','LineWidth',0.5);
    showShape(shape,bboxes(labels=='Truck',:),...
              'Color','magenta','LineWidth',0.5);
    showShape(shape,bboxes(labels=='Pedestrain',:),...
              'Color','yellow','LineWidth',0.5);
    hold off;
end

function [imageMap,ptCldOut] = preprocess(ptCld,gridParams, xMin, yMin)
    pcRange = [gridParams{1,1}{1} gridParams{1,1}{2} gridParams{1,1}{3} ...
               gridParams{1,1}{4} gridParams{1,1}{5} gridParams{1,1}{6}];

    indices = findPointsInROI(ptCld,pcRange);
    ptCldOut = select(ptCld,indices);

    bevHeight = gridParams{1,2}{2};
    bevWidth = gridParams{1,2}{1};

    gridH = gridParams{1,3}{2};
    gridW = gridParams{1,3}{1};

    % ZERO INTENSITY
    loc = ptCldOut.Location;
    if ~isfield(ptCld,'Intensity') || all(ptCld.Intensity == 0)
        intensity = zeros(size(ptCld.Location,1),1);
    else
        intensity = ptCld.Intensity;
    end

    loc(:,1) = int32(floor((loc(:,1) - xMin) / gridH)) + 1;
    loc(:,2) = int32(floor((loc(:,2) - yMin) / gridW)) + 1;

    loc(:,3) = loc(:,3) - min(loc(:,3));
    loc(:,3) = loc(:,3)/(pcRange(6) - pcRange(5));

    [~,I] = sortrows(loc,[1,2,-3]);
    locMod = loc(I,:); intensityMod = intensity(I,:);

    heightMap = zeros(bevHeight,bevWidth);
    intensityMap = zeros(bevHeight,bevWidth);

    locMod(:,1) = min(max(locMod(:,1),1), bevHeight);
    locMod(:,2) = min(max(locMod(:,2),1), bevWidth);

    mapIndices = sub2ind([bevHeight,bevWidth],locMod(:,1),locMod(:,2));
    [~,idx] = unique(mapIndices,"rows","first");

    binc = 1:bevWidth*bevHeight;
    counts = hist(mapIndices,binc);
    normalizedCounts = min(1.0, log(counts + 1) / log(64));

    for i = 1:size(idx,1)
        heightMap(mapIndices(idx(i))) = locMod(idx(i),3);
        intensityMap(mapIndices(idx(i))) = intensityMod(idx(i),1);
    end

    densityMap = reshape(normalizedCounts,[bevHeight,bevWidth]);

    % ZERO INTENSITY
    imageMap = zeros(bevHeight,bevWidth,3);
    imageMap(:,:,1) = densityMap;
    imageMap(:,:,2) = heightMap;
    imageMap(:,:,3) = heightMap;
end
