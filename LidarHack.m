outputFolder = fullfile(tempdir,'Pandaset');

lidarURL = ['https://ssd.mathworks.com/supportfiles/lidar/data/' ...
            'Pandaset_LidarData.tar.gz'];

helperDownloadPandasetData(outputFolder,lidarURL);

%Create a file datastore to load the PCD files from the specified path using the pcread function.
path = fullfile(outputFolder,'Lidar');
pcds = fileDatastore(path,'ReadFcn',@(x) pcread(x));

%Load the 3-D bounding box labels of the car, truck, and pedestrian objects.
gtPath = fullfile(outputFolder,'Cuboids','PandaSetLidarGroundTruth.mat');
data = load(gtPath,'lidarGtLabels');
Labels = timetable2table(data.lidarGtLabels);
boxLabels = Labels(:,2:end);

%Display the full-view point cloud
figure
ptCld = read(pcds);
ax = pcshow(ptCld.Location);
set(ax,'XLim',[-50 50],'YLim',[-40 40]);
zoom(ax,2.5);
axis off;

%The PandaSet data consists of full-view point clouds. For this example,
% crop the full-view point clouds and convert them to a bird's-eye-view images 
% using the standard parameters. 
% These parameters determine the size of the input passed to the network. 
% Selecting a smaller range of point clouds along the x-, y-, 
% and z-axes helps you detect objects that are closer to the origin.
xMin = -25.0;     
xMax = 25.0;      
yMin = 0.0;      
yMax = 50.0;      
zMin = -7.0;     
zMax = 15.0;   

%Define the dimensions for the bird's-eye-view image.
bevHeight = 608;
bevWidth = 608;

% find grid resolution 
gridW = (yMax - yMin)/bevWidth;
gridH = (xMax - xMin)/bevHeight;

% def grid parameters
gridParams = {{xMin,xMax,yMin,yMax,zMin,zMax},{bevWidth,bevHeight},{gridW,gridH}};

% Convert the training data to bird's-eye-view images by using the transformPCtoBEV helper function
% set writefiles to false if training data is already present
writeFiles = true;
if writeFiles
    transformPCtoBEV(pcds,boxLabels,gridParams,outputFolder);
end

%Use imageDatastore for loading the bird's-eye-view images.
dataPath = fullfile(outputFolder,'BEVImages');
imds = imageDatastore(dataPath);

%Use boxLabelDatastore for loading the ground truth boxes.
labelPath = fullfile(outputFolder,'Cuboids','BEVGroundTruthLabels.mat');
load(labelPath,'processedLabels');
blds = boxLabelDatastore(processedLabels);

%Remove the data that has no labels from the training data. 
% TODO: add back pcds
[imds,blds] = removeEmptyData(imds,blds);

%Split the data set into training, validation and test sets.
rng(0);
shuffledIndices = randperm(size(imds.Files,1));
idx = floor(0.6 * length(shuffledIndices));

trainingIdx = 1:idx;
validationIdx = idx+1 : (idx+1+floor(0.1*length(shuffledIndices)));
testIdx = validationIdx(end)+1 : length(shuffledIndices);

%Use imageDatastore and boxLabelDatastore to create datastores 
% for loading the image and label data during training and evaluation.
imdsTrain = subset(imds,shuffledIndices(trainingIdx));
bldsTrain = subset(blds,shuffledIndices(trainingIdx));

imdsValidation = subset(imds,shuffledIndices(validationIdx));
bldsValidation = subset(blds,shuffledIndices(validationIdx));

imdsTest = subset(imds,shuffledIndices(testIdx));
bldsTest = subset(blds,shuffledIndices(testIdx));

%Combine the image and box label datastores.
trainData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

% detect errors in data
validateInputDataComplexYOLOv4(trainData);
validateInputDataComplexYOLOv4(validationData);
validateInputDataComplexYOLOv4(testData);

% Preprocess the training data to prepare for training
networkInputSize = [608 608 3];
preprocessedTrainingData = transform(trainData,@(data)preprocessData(data,networkInputSize));

%Read the preprocessed training data.
data = read(preprocessedTrainingData);

%Display an image with the bounding boxes
I = data{1,1};
bbox = data{1,2};
labels = data{1,3};
helperDisplayBoxes(I,bbox,labels);

% Reset the datastore.
reset(preprocessedTrainingData);

% specify name and object class
classNames = {'Car'
              'Truck'
              'Pedestrain'};


% Strip the angle column (5th) from all label datastores
bldsTrain = transform(bldsTrain, @(data) {data{1}(:,1:4), data{2}});
bldsValidation = transform(bldsValidation, @(data) {data{1}(:,1:4), data{2}});
bldsTest = transform(bldsTest, @(data) {data{1}(:,1:4), data{2}});

% Recombine after cleaning
trainData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

% test for checking bboxes is displayed correctly
data = read(trainData);
I = data{1};
bboxes = data{2};
labels = data{3};

size(bboxes)
class(bboxes)

% Use the estimateAnchorBoxes function to estimate anchor 
% boxes based on the size of objects in the training data. 
rng(0)
numAnchors = 6;
[anchors,meanIoU] = estimateAnchorBoxes(trainData,numAnchors)

% Specify anchorBoxes to use in all the detection heads.
area = anchors(:, 1).*anchors(:,2);
[~,idx] = sort(area,"descend");

anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    };

% Create the YOLO v4 object detector by using the yolov4ObjectDetector function. 
modelName = "tiny-yolov4-coco";
detector = yolov4ObjectDetector(modelName,classNames,anchorBoxes,InputSize=networkInputSize);

% Use trainingOptions to specify network training options. 
options = trainingOptions("adam",...
    GradientDecayFactor=0.9,...
    SquaredGradientDecayFactor=0.999,...
    InitialLearnRate=0.001,...
    LearnRateSchedule="none",...
    MiniBatchSize=4,...
    L2Regularization=0.0005,...
    MaxEpochs=50,...
    BatchNormalizationStatistics="moving",...
    DispatchInBackground=true,...
    ResetInputNormalization=false,...
    Shuffle="every-epoch",...
    VerboseFrequency=100,...
    ValidationFrequency=1000,...
    CheckpointPath=tempdir,...
    ValidationData=validationData);

% Train YOLO v4 object detector.
doTraining = false;

% Convert yolov4ObjectDetector to ONNX
% exportONNXNetwork(detector.Network,'trainedYOLOv4_raw.onnx');

if doTraining
    [detector,info] = trainYOLOv4ObjectDetector(trainData,detector,options);
else
    % Load pretrained detector for the example.
    detector = downloadPretrainedComplexYOLOv4(modelName);
end

save('trainedYOLOv4.mat','detector');

% Evaluate Model
% Reset the datastore.
%reset(testData)
% Run the detector on images in the test set and collect the results.
%results = detect(detector,testData,'ExecutionEnvironment','cpu');

% Evaluate the object detector using the average precision metric.
%metrics = evaluateObjectDetection(results,testData,'AdditionalMetrics','AOS');
%metrics.ClassMetrics

% Detect Objects Using Trained Complex-YOLO V4
% Read the datastore.
reset(testData)

% Read the BEV image from the test data.
data = read(testData);
I = data{1,1};

% Run the detector.
[bboxes,scores,labels] = detect(detector,I);

% Display the output.
figure
helperDisplayBoxes(I,bboxes,labels);

%Transfer the detected boxes to a point cloud
lidarTestData = subset(pcds,shuffledIndices(testIdx));
ptCld = read(lidarTestData);
[ptCldOut,bboxCuboid] = transferbboxToPointCloud(bboxes,gridParams,ptCld);
helperDisplayBoxes(ptCldOut,bboxCuboid,labels);



% supporting functions

%preprocess data
function data = preprocessData(data,targetSize)
% Resize the images and scale the pixels to between 0 and 1. Also scale the
% corresponding bounding boxes.
for ii = 1:size(data,1)
    I = data{ii,1};
    imgSize = size(I);
    
    % Convert an input image with a single channel to three channels.
    if numel(imgSize) < 3 
        I = repmat(I,1,1,3);
    end
    bboxes = data{ii,2};

    I = im2single(imresize(I,targetSize(1:2)));
    scale = targetSize(1:2)./imgSize(1:2);
    bboxes = bboxresize(bboxes,scale);   
    data(ii, 1:2) = {I,bboxes};
end
end


%Utility Functions
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
    showShape(shape,bboxes(labels=='Car',:),...
                  'Color','green','LineWidth',0.5);hold on;
    showShape(shape,bboxes(labels=='Truck',:),...
              'Color','magenta','LineWidth',0.5);
    showShape(shape,bboxes(labels=='Pedestrain',:),...
              'Color','yellow','LineWidth',0.5);
    hold off;
end

function helperDownloadPandasetData(outputFolder,lidarURL)
% Download the data set from the given URL to the output folder.
    lidarDataTarFile = fullfile(outputFolder,'Pandaset_LidarData.tar.gz');    
    if ~exist(lidarDataTarFile,'file')
        mkdir(outputFolder);        
        disp('Downloading PandaSet Lidar driving data (5.2 GB)...');
        websave(lidarDataTarFile,lidarURL);
        untar(lidarDataTarFile,outputFolder);
    end    
    % Extract the file.
    if (~exist(fullfile(outputFolder,'Lidar'),'dir'))...
            &&(~exist(fullfile(outputFolder,'Cuboids'),'dir'))
        untar(lidarDataTarFile,outputFolder);
    end
end


function transformPCtoBEV(lidarData,boxLabels,gridParams,dataLocation)
% createBEVData create the Bird's-Eye-View image data adn the corresponding
% labels from the given dataset.
%
% Copyright 2021 The MathWorks, Inc.

% Get classnames of Pandaset dataset.
classNames = boxLabels.Properties.VariableNames;

% Get the number of files.
numFiles = size(boxLabels,1);
processedLabels = cell(size(boxLabels));

% Reset the point cloud datastore.
reset(lidarData);

 for i = 1:numFiles
     
    ptCloud = read(lidarData);     
    groundTruth = boxLabels(i,:);

    [processedData,~] = preprocess(ptCloud,gridParams);

    for ii = 1:numel(classNames)
        labels = groundTruth(1,classNames{ii}).Variables;
        processedLabels{i,ii} = [];
        if(iscell(labels))
            labels = labels{1};
        end
        if ~isempty(labels)

            % Get the label indices that are in the selected RoI.
            labelsIndices = labels(:,1) - labels(:,4) > gridParams{1,1}{1} ...
                          & labels(:,1) + labels(:,4) < gridParams{1,1}{2} ...
                          & labels(:,2) - labels(:,5) > gridParams{1,1}{3} ...
                          & labels(:,2) + labels(:,5) < gridParams{1,1}{4} ...
                          & labels(:,4) > 0 ...
                          & labels(:,5) > 0 ...
                          & labels(:,6) > 0;
            labels = labels(labelsIndices,:);

            labelsBEV = labels(:,[2,1,5,4,9]);
            labelsBEV(:,5) = -labelsBEV(:,5);

            labelsBEV(:,1) = int32(floor(labelsBEV(:,1)/gridParams{1,3}{1})) + 1;
            labelsBEV(:,2) = int32(floor(labelsBEV(:,2)/gridParams{1,3}{2})+gridParams{1,2}{2}/2) + 1;

            labelsBEV(:,3) = int32(floor(labelsBEV(:,3)/gridParams{1,3}{1})) + 1;
            labelsBEV(:,4) = int32(floor(labelsBEV(:,4)/gridParams{1,3}{2})) + 1;
            processedLabels{i,ii} = labelsBEV;
        end
        
    end
    
    writePath = fullfile(dataLocation,'BEVImages');
    if ~isfolder(writePath)
        mkdir(writePath);
    end
    
    imgSavePath = fullfile(writePath,sprintf('%04d.jpg',i));
    imwrite(processedData,imgSavePath);

end

processedLabels = cell2table(processedLabels);
numClasses = size(processedLabels,2);
for j = 1:numClasses
    processedLabels.Properties.VariableNames{j} = classNames{j};
end

labelsSavePath = fullfile(dataLocation,'Cuboids/BEVGroundTruthLabels.mat');
save(labelsSavePath,'processedLabels');
end

%% Get the BEV image from point cloud.

function [imageMap,ptCldOut] = preprocess(ptCld,gridParams)

    pcRange = [gridParams{1,1}{1} gridParams{1,1}{2} gridParams{1,1}{3} ...
               gridParams{1,1}{4} gridParams{1,1}{5} gridParams{1,1}{6}]; 

    indices = findPointsInROI(ptCld,pcRange);
    ptCldOut = select(ptCld,indices);
    
    bevHeight = gridParams{1,2}{2};
    bevWidth = gridParams{1,2}{1};
    
    % Find grid resolution.
    gridH = gridParams{1,3}{2};
    gridW = gridParams{1,3}{1};
    
    loc = ptCldOut.Location;
    %intensity = ptCldOut.Intensity;
    %intensity = normalize(intensity,'range');
    % ZERO INTENSITY
    %if ~isfield(ptCld,'Intensity') || all(ptCld.Intensity == 0)
    %    intensity = zeros(size(ptCld.Location,1),1);
    %else
    %    intensity = ptCld.Intensity;
    %end
    intensity = 0;
    
    % Find the grid each point falls into.
    loc(:,1) = int32(floor(loc(:,1)/gridH)+bevHeight/2) + 1;
    loc(:,2) = int32(floor(loc(:,2)/gridW)) + 1;
    
    % Normalize the height.
    loc(:,3) = loc(:,3) - min(loc(:,3));
    loc(:,3) = loc(:,3)/(pcRange(6) - pcRange(5));
    
    % Sort the points based on height.
    [~,I] = sortrows(loc,[1,2,-3]);
    locMod = loc(I,:);
    intensityMod = intensity(I,:);
    
    % Initialize height and intensity map
    heightMap = zeros(bevHeight,bevWidth);
    intensityMap = zeros(bevHeight,bevWidth);
    
    locMod(:,1) = min(locMod(:,1),bevHeight);
    locMod(:,2) = min(locMod(:,2),bevHeight);
    
    % Find the unique indices having max height.
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
    
    % because ZERO INTENSITY
    imageMap = zeros(bevHeight,bevWidth,3);
    imageMap(:,:,1) = densityMap;       % R channel
    imageMap(:,:,2) = heightMap;        % G channel
    %imageMap(:,:,3) = intensityMap;     % B channel
    imageMap(:,:,3) = heightMap;
end

function [imdsProcessed,bdsProcessed] = removeEmptyData(imds,bds)
% Return non-empty indices from the saved data

% Copyright 2021 The MathWorks, Inc.

% Read labels from the box label datastore.
processedLabels = readall(bds);

% Get the non-empty indices.
indices = ~cellfun('isempty',processedLabels(:,1));

imdsProcessed = subset(imds,indices);
bdsProcessed = subset(bds,indices);

end

function validateInputDataComplexYOLOv4(ds)
% Validates the input images, bounding boxes and labels and displays the 
% paths of invalid samples. 

% Copyright 2021 The MathWorks, Inc.

% Path to images
info = ds.UnderlyingDatastores{1}.Files;

ds = transform(ds, @isValidDetectorData);
data = readall(ds);

validImgs = [data.validImgs];
validBoxes = [data.validBoxes];
validLabels = [data.validLabels];

msg = "";

if(any(~validImgs))
    imPaths = info(~validImgs);
    str = strjoin(imPaths, '\n');
    imErrMsg = sprintf("Input images must be non-empty and have 2 or 3 dimensions. The following images are invalid:\n") + str;
    msg = (imErrMsg + newline + newline);
end

if(any(~validBoxes))
    imPaths = info(~validBoxes);
    str = strjoin(imPaths, '\n');
    boxErrMsg = sprintf("Bounding box data must be M-by-5 matrices of positive integer values. The following images have invalid bounding box data:\n") ...
        + str;
    
    msg = (msg + boxErrMsg + newline + newline);
end

if(any(~validLabels))
    imPaths = info(~validLabels);
    str = strjoin(imPaths, '\n');
    labelErrMsg = sprintf("Labels must be non-empty and categorical. The following images have invalid labels:\n") + str;
    
    msg = (msg + labelErrMsg + newline);
end

if(~isempty(msg))
    error(msg);
end

end

function out = isValidDetectorData(data)
% Checks validity of images, bounding boxes and labels
for i = 1:size(data,1)
    I = data{i,1};
    boxes = data{i,2};
    labels = data{i,3};

    imageSize = size(I);
    mSize = size(boxes, 1);

    out.validImgs(i) = iCheckImages(I);
    out.validBoxes(i) = iCheckBoxes(boxes, imageSize);
    out.validLabels(i) = iCheckLabels(labels, mSize);
end

end

function valid = iCheckImages(I)
% Validates the input images.

valid = true;
if ndims(I) == 2
    nDims = 2;
else
    nDims = 3;
end
% Define image validation parameters.
classes        = {'numeric'};
attrs          = {'nonempty', 'nonsparse', 'nonnan', 'finite', 'ndims', nDims};
try
    validateattributes(I, classes, attrs);
catch
    valid = false;
end
end

function valid = iCheckBoxes(boxes, imageSize)
% Validates the ground-truth bounding boxes to be non-empty and finite.

valid = true;
% Define bounding box validation parameters.
classes = {'numeric'};
attrs   = {'nonempty', 'nonnan', 'finite', 'positive', 'nonzero', 'nonsparse', '2d', 'ncols', 4};
attrsYaw = {'nonempty', 'nonnan', 'finite', 'nonsparse'};
try
    validateattributes(boxes(:,1:4), classes, attrs);
    validateattributes(boxes(:,5), classes, attrsYaw);
    % Validate if bounding box in within image boundary.
    validateattributes(boxes(:,1)+boxes(:,3)-1, classes, {'<=', imageSize(2)});
    validateattributes(boxes(:,2)+boxes(:,4)-1, classes, {'<=', imageSize(1)}); 
catch
    valid = false;
end
end

function valid = iCheckLabels(labels, mSize)
% Validates the labels.

valid = true;
% Define label validation parameters.
classes = {'categorical'};
attrs   = {'nonempty', 'nonsparse', '2d', 'ncols', 1, 'nrows', mSize};
try
    validateattributes(labels, classes, attrs);
catch
    valid = false;
end
end