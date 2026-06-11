% Generate a Walker Delta constellation with STK, then keep the original
% station import, satellite position export, satellite renaming, and
% visibility calculation workflow.
%
% Walker Tool equivalent:
%   Type                     = Delta
%   Number of Sats per Plane = satsPerPlane
%   Number of Planes         = numPlanes
%   Inter Plane Spacing      = phaseFactor
%   RAAN Spread              = 360 deg

USE_ENGINE = false;

% Initialize STK.
if USE_ENGINE
    app = actxserver('STKX11.application');
    root = actxserver('AgStkObjects11.AgStkObjectRoot');
else
    app = actxserver('STK11.application');
    root = app.Personality2;
end

% Scenario time.
StartTime = '6 Jan 2025 00:00:00.000';
StopTime  = '6 Jan 2025 00:10:00.000';

scenario = root.Children.New('eScenario', 'MATLAB_PredatorMission');
scenario.SetTimePeriod(StartTime, StopTime);
scenario.StartTime = StartTime;
scenario.StopTime = StopTime;

if USE_ENGINE
    % STK Engine mode.
else
    try
        root.ExecuteCommand('Animate * Reset');
        disp('Animate reset finished.');
    catch ME
        disp('Animate reset failed:');
        disp(ME.message);
    end
end

% ====================== Read ground stations ======================
folder = 'C:\usrspace\stkfile\position\stationfile';

files = dir(fullfile(folder, '*.txt'));
numFiles = length(files);

file_IDs = zeros(numFiles, 1);
for k = 1:numFiles
    [~, nameNoExt, ~] = fileparts(files(k).name);
    file_IDs(k) = str2double(nameNoExt);
end

[sorted_IDs, sortIndex] = sort(file_IDs);
sortedFiles = files(sortIndex);

stations = struct('id', {}, 'position', {}, 'angle', {});

disp('Start reading stations...');

for k = 1:numFiles
    thisFile = sortedFiles(k);
    fullFileName = fullfile(folder, thisFile.name);
    currentID = sorted_IDs(k);

    xyz_data = load(fullFileName);

    if size(xyz_data, 1) > 1
        pos = xyz_data(1, 1:3);
    else
        pos = xyz_data(1, 1:3);
    end

    stations(currentID).id = currentID;
    stations(currentID).position = pos;
    stations(currentID).angle = deg2rad(20);

    fprintf('Stored stations(%d): file %s, position [%.1f, ...]\n', ...
        currentID, thisFile.name, pos(1));
end

disp('Station reading finished.');

% ====================== Walker Delta parameters ======================
% Starlink-like example from the literature:
%   totalSatellites / numPlanes / phaseFactor = 1584 / 72 / 1
%   altitude = 540 km or 550 km depending on the selected source.
numPlanes = 72;
satsPerPlane = 22;
totalSatellites = numPlanes * satsPerPlane;
phaseFactor = 1;

height = 540;          % km
inclination = 53.2;    % deg
baseRaan = 0;          % deg
baseAnomaly = 0;       % deg

timestep = 1;

fprintf('Walker Delta: T/P/F = %d/%d/%d, satsPerPlane = %d\n', ...
    totalSatellites, numPlanes, phaseFactor, satsPerPlane);
fprintf('Orbit: altitude = %.3f km, inclination = %.3f deg\n', ...
    height, inclination);

% ====================== Output and helper objects ======================
baseOutDir = 'C:\usrspace\stkfile\position\oneperiod';
if ~exist(baseOutDir, 'dir')
    mkdir(baseOutDir);
end

export = module.Export_Position_STK();
read_file = module.Read_All_E_Files_XYZ();
satObj = module.sat();

% ====================== Create Walker Delta constellation ======================
% This block is equivalent to STK Walker Tool:
%   Type = Delta
%   Number of Sats per Plane = satsPerPlane
%   Number of Planes = numPlanes
%   Inter Plane Spacing = phaseFactor
%   RAAN Spread = 360 deg
%
% The seed satellite provides h, i, RAAN and initial anomaly. STK creates
% all planes and applies the Walker Delta phasing internally.
seedSatelliteName = 'WalkerDeltaSeed';

params = struct();
params.satelliteName = seedSatelliteName;
params.perigeeAlt = height;
params.apogeeAlt = height;
params.inclination = inclination;
params.argOfPerigee = 0;
params.RAAN = baseRaan;
params.Anomaly = baseAnomaly;

satObj.createSatellite(root, scenario, params);

params_constellation = struct();
params_constellation.seedSatelliteName = seedSatelliteName;
params_constellation.numPlanes = numPlanes;
params_constellation.numSatsPerPlane = satsPerPlane;
params_constellation.interPlanePhaseIncrement = phaseFactor;

satObj.createWalkerConstellation_Delta(root, params_constellation);

% Keep the original behavior: unload the seed satellite after Walker creation.
unloadCmd = sprintf('Unload / */Satellite/%s', seedSatelliteName);
root.ExecuteCommand(unloadCmd);

% ====================== Rename satellites ======================
sat = module.sat();

satellite_names = sat.getSatelliteNames(scenario);
sat.batchRenameSatellitesInSTK2(root, satellite_names);

satellite_names = sat.getSatelliteNames(scenario);
numberofsatellite = length(satellite_names);

fprintf('Satellite count after Walker creation and seed unload: %d\n', ...
    numberofsatellite);

if numberofsatellite ~= totalSatellites
    warning('Expected %d satellites, but STK currently has %d satellites.', ...
        totalSatellites, numberofsatellite);
end

% ====================== Export satellite positions ======================
dayTag = sprintf('baseRaan_%d', 1);

outDir = fullfile(baseOutDir, dayTag);
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

output_folder = fullfile(outDir, 'satellite_pos');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

export.Export_Pos_Line_J2000(root, scenario, output_folder);

root.ExecuteCommand('UnloadMulti / */Satellite/*');

% ====================== Read positions and calculate visibility ======================
XYZ = read_file.Read_All_E_Files_XYZ_Serial(output_folder);

output_file_dir = fullfile(outDir, ['station_visible_satellites_' dayTag '.xml']);

Station_View_Result = module.Calculate_Constellation_Visibility_para2( ...
    stations, XYZ, output_file_dir);

clear XYZ Station_View_Result;

% ====================== End ======================
