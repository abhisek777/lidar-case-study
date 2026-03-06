function Processing_LIDAR_Example(folder_data_in)
%% Processing_LIDAR_Example - Gives an example how to read and display provided lidar data in *.csv
% Syntax:  Processing_LIDAR_Example(folder_data_in)
%
% Inputs:
%    folder_data_in - gives the path and folder containing the data
%
% Outputs:
%    no outputs
%
% Example: 
%    folder_data_in = 'E:\Blickfeld_LiDAR\data\Data_Street'
%    Processing_LIDAR_Example(folder_data_in)
%
% Other m-files required: read_blickfeld_csv.m
% Subfunctions: none
% MAT-files required: none
% MAT-Toolbox required: Computer Vision Toolbox
%
% See also: none
% Author: Lehmann, Benjamin 
% IU Internationale Hochschule
% email: b.lehmann.ext@iubh-fernstudium.de
% Website: http://www.iu.de
% April 2021; Last revision: 28-April-2021
%% ------------- CODE --------------

data_files = dir([folder_data_in,filesep,'*.csv']);

for k = 1 : numel(data_files)
    [ptsc,point_id, ambient, timestamp] = read_blickfeld_csv([folder_data_in, filesep, data_files(k).name]);
    figure(1)
    pcshow(ptsc);
    axis([-25 25 0 60 0 30])
    set(gca,'View',[ -32   35])
    xlabel('x in m'); ylabel('y in m');zlabel('z in m')
    title(['file: ',data_files(k).name],'Interpreter','none')
    pause(1)
end

end