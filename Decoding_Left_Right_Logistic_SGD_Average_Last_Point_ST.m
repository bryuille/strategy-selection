function [] = Decoding_Left_Right_Logistic_SGD_Average_Last_Point_ST(PROBE, MONKEY)



addpath(genpath('/om2/user/mramadan/Neurophys/Sorting/'));


data_directory= strcat('/om2/user/mramadan/Neurophys/Sorting/Analysis/', PROBE, '/', MONKEY, '/', 'Good_Trials', '/'); 

FR_directory = strcat('/om2/user/mramadan/Neurophys/Sorting/Analysis/', PROBE, '/' , MONKEY, '/Firing_Rates/');

om4_directory = strcat('/om4/group/jazlab/Mahdi/Neurophys/Sorting/Sorted_Data/',PROBE,'/',MONKEY,'/');

save_directory_PCA = strcat('/om2/user/mramadan/Neurophys/Sorting/Analysis/', PROBE, '/', MONKEY, '/Decoding_Figs/Left_Right_Flash_One_To_Flash_Three_Plus_300/Logistic_SGD_Last_Point/');

num_sess = 0; 

files = dir(om4_directory);
directoryNames = {files.name};
directoryNames = directoryNames(~ismember(directoryNames,{'.','..','.DS_Store'}));

format long


dr = rgb('deep red');
pin = rgb('salmon');

cy =  rgb('cyan');
dbl = rgb('dark blue');

gld = rgb('gold');
dor = rgb('dark orange');




first_len= [500, 500, 1000, 1000, 500, 500, 1000, 1000, 500, 500, 750, 750, 500, 500, 750, 750, 500, 500, 500, 500, 500, 500, 500, 500];
second_len =  [900, 1100, 700,	500, 700, 1100, 900, 500, 900, 1100, 700, 500, 700, 1100, 900, 500, 900, 1100, 700, 500, 700, 1100, 900,500];


stop_len = [1700, 1900, 2000, 1800, 1500, 1900, 2200, 1800, 1700, 1900, 1750, 1550, 1500, 1900, 1950, 1550, 1700, 1900, 1500, 1300, 1500, 1900, 1700, 1300]; 
cond_len = [0, 1700, 1900, 2000, 1800, 1500, 1900, 2200, 1800, 1700, 1900, 1750, 1550, 1500, 1900, 1950, 1550, 1700, 1900, 1500, 1300, 1500, 1900, 1700, 1300];

% PERFORMANCE_THRESHOLD = 85;

decoding_performance_all = cell(1,24);
decoding_L_R_all = cell(1,24);

decoding_performance_shuffle_all = cell(1,24);
decoding_L_R_shuffle_all = cell(1,24);

correct_incorrect_all = cell(1,24);

decoding_performance_all = cell(1,24);
decoding_L_R_all = cell(1,24);


session_performances = [];
session_performances_shuffle = [];

% Loop through each directory name to compute performance
for i = 1:length(directoryNames)

    % Check if the performance file exists
    if isfile(strcat(save_directory_PCA, '_decoding_data_performance_', directoryNames{i}, '.mat'))

        % Load the required data (ACTUAL)
        decoding_performance_load = load(strcat(save_directory_PCA, '_decoding_data_performance_', directoryNames{i}, '.mat'));
        decoding_performance = decoding_performance_load.model_pred_accuracy;

        % Calculate the performance for this session (ACTUAL)
        session_perf = nanmean(cellfun(@(x) nanmean(x(:, end)), decoding_performance)) * 100;

        % Append to performance array (ACTUAL)
        session_performances = [session_performances; session_perf];

        % --- NEW: also load SHUFFLE (if exists), keep aligned by session order ---
        if isfile(strcat(save_directory_PCA, '_decoding_data_performance_shuffle_', directoryNames{i}, '.mat'))
            decoding_performance_shuffle_load = load(strcat(save_directory_PCA,'_decoding_data_performance_shuffle_',directoryNames{i},'.mat'));
            decoding_performance_shuffle = decoding_performance_shuffle_load.model_pred_accuracy_shuffle;

            session_perf_shuffle = nanmean(cellfun(@(x) nanmean(x(:, end)), decoding_performance_shuffle)) * 100;
            session_performances_shuffle = [session_performances_shuffle; session_perf_shuffle];
        else
            % keep alignment: put NaN if shuffle missing
            session_performances_shuffle = [session_performances_shuffle; NaN];
        end
    end
end

% Sort the performances in descending order
sorted_performances = sort(session_performances, 'descend');

% Select the 10th-best performance as the threshold (robust if <10 sessions)
idx10 = min(10, numel(sorted_performances));
PERFORMANCE_THRESHOLD = sorted_performances(idx10);

% NEW: mask for top sessions (based on ACTUAL), and require shuffle present for plotting
top_mask = (session_performances >= PERFORMANCE_THRESHOLD);
top_mask = top_mask & ~isnan(session_performances_shuffle);

top10_session_performances        = session_performances(top_mask);
top10_session_performances_shuffle = session_performances_shuffle(top_mask);

% (Optional) if ties gave you >10, force exactly 10 highest ACTUAL among those with shuffle
if numel(top10_session_performances) > 10
    [~, ord] = sort(top10_session_performances, 'descend');
    ord = ord(1:10);
    top10_session_performances         = top10_session_performances(ord);
    top10_session_performances_shuffle = top10_session_performances_shuffle(ord);
end

% ---------------- Violin plot: TOP-10 Actual vs Shuffle (0–100) ----------------
figure('Position',[200 200 600 400]);

V = [top10_session_performances(:), top10_session_performances_shuffle(:)];  % n×2
violinplot(V, [], ...
    'ViolinAlpha',0.2, ...
    'MarkerSize',10, ...
    'ShowMean',true);

% xlim([0.5 2.5]);   % <-- remove this
xticklabels({'Actual (Top 10)','Shuffle (Top 10)'});
ylabel('Decoding accuracy at last timepoint (%)');
ylim([0 100]);
set(gca,'YLimMode','manual');
title(sprintf('Overall decoding performance (Top 10; n=%d sessions)', size(V,1)));

savefig(gcf, fullfile(save_directory_PCA, 'All_Sessions_Decoding_ST_Top10_Performance_Violin.fig'));
% -------------------------------------------------------------------------------




for i = 1:length(directoryNames)

if isfile (strcat(save_directory_PCA,'_decoding_data_performance_',directoryNames{i},'.mat'))
 
 display(directoryNames{i})
 
 decoding_L_R_load = load(strcat(save_directory_PCA,'_model_pred_L_R_trials_',directoryNames{i},'.mat'));
  
 decoding_perf_within_load = load(strcat(save_directory_PCA,'_decoding_perf_within_',directoryNames{i},'.mat'));

 correct_incorrect_load = load(strcat(save_directory_PCA,'_IC_answer_train_',directoryNames{i},'.mat'));

 decoding_L_R_shuffle_load = load(strcat(save_directory_PCA,'_model_pred_L_R_trials_shuffle_',directoryNames{i},'.mat'));

 decoding_performance_load = load(strcat(save_directory_PCA,'_decoding_data_performance_',directoryNames{i},'.mat'));

 decoding_performance_shuffle_load = load(strcat(save_directory_PCA,'_decoding_data_performance_shuffle_',directoryNames{i},'.mat'));

 decoding_L_R = decoding_L_R_load.model_pred_L_R_trials;
 decoding_perf_within = decoding_perf_within_load.best_decoding_t_value;
 correct_incorrect= correct_incorrect_load.IC_answer_train;
 decoding_L_R_shuffle = decoding_L_R_shuffle_load.model_pred_L_R_trials_shuffle;
 decoding_performance = decoding_performance_load.model_pred_accuracy;
 decoding_performance_shuffle = decoding_performance_shuffle_load.model_pred_accuracy_shuffle;

 overall_decoder_accur = [];

 % if decoding_perf_within > PERFORMANCE_THRESHOLD
 nanmean(cellfun(@(x) nanmean(x(:, end)), decoding_performance))*100
 
 if nanmean(cellfun(@(x) nanmean(x(:, end)), decoding_performance))*100 >= PERFORMANCE_THRESHOLD

     num_sess = num_sess + 1;

     for conds = 1:24
     
    
            decoding_L_R_all{conds} = vertcat(decoding_L_R_all{conds}, nanmean(decoding_L_R{conds},3));  
            correct_incorrect_all{conds}  = vertcat(correct_incorrect_all{conds}, squeeze(correct_incorrect{conds}(:,:,1)));

            decoding_L_R_shuffle_all{conds} = vertcat(decoding_L_R_shuffle_all{conds},nanmean(decoding_L_R_shuffle{conds},3));

            decoding_performance_all{conds} = vertcat(decoding_performance_all{conds},nanmean(decoding_performance{conds}(:,1:end,:),1) );
            decoding_performance_shuffle_all{conds} = vertcat(decoding_performance_shuffle_all{conds},nanmean(decoding_performance_shuffle{conds}(:,1:end,:),1) );
          
     end

 end


end

end

% -------------------------------------------------------------------------
% Helper (unchanged)
function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end

% -------------------------------------------------------------------------
%  MAIN PLOTTING LOOP  (drop-in replacement)
% -------------------------------------------------------------------------

% ---------- colour anchors ------------------------------------------------
dr  = rgb('deep red');     pin = rgb('salmon');      % mazes 1–2
cy  = rgb('cyan');         dbl = rgb('dark blue');   % mazes 3–4
gld = rgb('gold');         dor = rgb('dark orange'); % mazes 5–6

pair_lohi = { {dr,  pin}, {cy,  dbl}, {gld, dor} };  % dark , light

ls_actual  = {'-','--'};    % solid / dashed    (actual)
ls_shuffle = {':','-.'};    % dotted / dash-dot (shuffle)
ls_flash   = {'-','--'};    % flash lines: first path solid, second dashed

Tmax = max(stop_len);

for maze = 1:6
    % ------- 2-color mapping: (1,2)=first; (3,4)=second --------------------
    set_idx   = ceil(maze/2);                  % 1,1,2,2,3,3
    % first color, second color per maze pair:
    pair_firstsecond = { {pin, dr}, {cy, dbl}, {gld, dor} };
    c_first  = pair_firstsecond{set_idx}{1};   % for paths 1 & 2
    c_second = pair_firstsecond{set_idx}{2};   % for paths 3 & 4
    
    % colours(path,:) used below
    colours = zeros(4,3);
    colours(1,:) = c_first;
    colours(2,:) = c_first;
    colours(3,:) = c_second;
    colours(4,:) = c_second;

    % ---------- CORRECT (0) then ERROR (1) --------------------------------
    for is_err = [0 1]                 % 0 = Correct, 1 = Error
        figure('Position',[100 100 300 1200]);
        sgtitle(sprintf('Maze %d – %s trials', ...
               maze, ternary(~is_err,'CORRECT','ERROR')), ...
               'FontWeight','bold');

        % ---------------- TOP subplot : paths 1 & 3 -----------------------
        subplot(2,1,1); hold on
        paths_top = [1 3];
        for k = 1:2
            path  = paths_top(k);
            cond  = (maze-1)*4 + path;

            flag  = (correct_incorrect_all{cond} == is_err); % 0→correct
            data  = decoding_L_R_all{cond}(flag,:);
            dataS = decoding_L_R_shuffle_all{cond}(flag,:);

            if ~isempty(dataS)   % shuffle (black)
                cols = 1:(size(dataS,2)-1);     % columns to keep   (1 … end-1)
                
                m = nanmean( dataS(:, cols), 1 );          % shuffle mean
                e = 0.5 * nanstd( dataS(:, cols), 0, 1 );   % shuffle SEM
                shadedErrorBar(cols, m, e, ...
                    'lineprops',{'Color','k','LineStyle',ls_shuffle{k}}, ...
                    'patchSaturation',0.05);
            end
            if ~isempty(data)    % actual
                cols = 1:(size(data,2)-1);      % same idea
                m = nanmean( data(:, cols), 1 );
                e = 0.5 * nanstd( data(:, cols), 0, 1 );
                shadedErrorBar(cols, m, e, ...
                    'lineprops',{'Color',colours(path,:),'LineStyle',ls_actual{k}}, ...
                    'patchSaturation',0.075);
            end
        end

        % flash markers for BOTH paths in this subplot
        for k = 1:2
            cond_k = (maze-1)*4 + paths_top(k);
            xline(first_len(cond_k),                  'Color','r','LineWidth',2,'LineStyle',ls_flash{k});
            xline(first_len(cond_k)+second_len(cond_k),'Color','r','LineWidth',2,'LineStyle',ls_flash{k});
        end

        xlim([0 Tmax]); ylim([0 1]);
        view(-90,90);                % decoding horizontal, time ↑
        set(gca,'YDir','reverse');   % flip vertical time axis → time ↓
        xlabel('Decoding (0 ↔ 1)'); ylabel('time (ms)');
        title('Paths 1 & 3 (time ↑)');

        % ---------------- BOTTOM subplot : paths 2 & 4 --------------------
        subplot(2,1,2); hold on
        paths_bot = [2 4];
        for k = 1:2
            path  = paths_bot(k);
            cond  = (maze-1)*4 + path;

            flag  = (correct_incorrect_all{cond} == is_err);
            data  = decoding_L_R_all{cond}(flag,:);
            dataS = decoding_L_R_shuffle_all{cond}(flag,:);

            if ~isempty(dataS)
                cols = 1:(size(dataS,2)-1);     % columns to keep   (1 … end-1)
                
                m = nanmean( dataS(:, cols), 1 );          % shuffle mean
                e = 0.5 * nanstd( dataS(:, cols), 0, 1 );   % shuffle SEM
                shadedErrorBar(cols, m, e, ...
                    'lineprops',{'Color','k','LineStyle',ls_shuffle{k}}, ...
                    'patchSaturation',0.05);
            end
            if ~isempty(data)
                cols = 1:(size(data,2)-1);      % same idea
                m = nanmean( data(:, cols), 1 );
                e = 0.5 * nanstd( data(:, cols), 0, 1 );
                shadedErrorBar(cols, m, e, ...
                    'lineprops',{'Color',colours(path,:),'LineStyle',ls_actual{k}}, ...
                    'patchSaturation',0.075);
            end
        end

        % flash markers for BOTH paths in this subplot
        for k = 1:2
            cond_k = (maze-1)*4 + paths_bot(k);
            xline(first_len(cond_k),                  'Color','r','LineWidth',2,'LineStyle',ls_flash{k});
            xline(first_len(cond_k)+second_len(cond_k),'Color','r','LineWidth',2,'LineStyle',ls_flash{k});
        end

        xlim([0 Tmax]); ylim([0 1]);
        view(-90,90);                % decoding horizontal
        set(gca,'XDir','reverse');   % flip vertical time axis → time ↓
        set(gca,'YDir','reverse');   % flip vertical time axis → time ↓
        xlabel('Decoding (0 ↔ 1)'); ylabel('time (ms)');
        title('Paths 2 & 4 (time ↓)');

        % ---------------- save & close ------------------------------------
        fname = sprintf('%sMaze_%d_%s', ...
                save_directory_PCA, maze, ternary(~is_err,'Correct','Error'));
        savefig(gcf, [fname '.fig']);
        close(gcf);
    end
end



% -------------------------------------------------------------------------
%  MAIN PLOTTING LOOP  (ALL trials pooled: Correct + Error combined)
%  Saves: All_Sessions_Decoding_ST_Maze_*
% -------------------------------------------------------------------------

% ---------- colour anchors ------------------------------------------------
dr  = rgb('deep red');     pin = rgb('salmon');      % mazes 1–2
cy  = rgb('cyan');         dbl = rgb('dark blue');   % mazes 3–4
gld = rgb('gold');         dor = rgb('dark orange'); % mazes 5–6

% 2-color mapping per maze pair (paths 1&2 vs paths 3&4)
pair_firstsecond = { {pin, dr}, {cy, dbl}, {gld, dor} };

ls_actual  = {'-','--'};    % solid / dashed    (actual) for the 2 lines in subplot
ls_shuffle = {':','-.'};    % dotted / dash-dot (shuffle) for the 2 lines in subplot
ls_flash   = {'-','--'};    % flash lines: first path solid, second dashed (within subplot)

Tmax = max(stop_len);

for maze = 1:6

    set_idx   = ceil(maze/2);                  % 1,1,2,2,3,3
    c_first   = pair_firstsecond{set_idx}{1};  % for paths 1 & 2
    c_second  = pair_firstsecond{set_idx}{2};  % for paths 3 & 4

    colours = zeros(4,3);
    colours(1,:) = c_first;
    colours(2,:) = c_first;
    colours(3,:) = c_second;
    colours(4,:) = c_second;

    figure('Position',[100 100 300 1200]);
    sgtitle(sprintf('Maze %d – All trials (Correct + Error pooled)', maze));

    % ================= TOP subplot : paths 1 & 3 ==========================
    subplot(2,1,1); hold on
    paths_top = [1 3];

    for k = 1:2
        path  = paths_top(k);
        cond  = (maze-1)*4 + path;             % maze1:1-4, maze2:5-8, ...

        data  = decoding_L_R_all{cond};
        dataS = decoding_L_R_shuffle_all{cond};

        if ~isempty(dataS)
            cols = 1:(size(dataS,2)-1);
            m = nanmean(dataS(:, cols), 1);
            e = 0.5 * nanstd(dataS(:, cols), 0, 1);
            shadedErrorBar(cols, m, e, ...
                'lineprops',{'Color','k','LineStyle',ls_shuffle{k}}, ...
                'patchSaturation',0.05);
        end

        if ~isempty(data)
            cols = 1:(size(data,2)-1);
            m = nanmean(data(:, cols), 1);
            e = 0.5 * nanstd(data(:, cols), 0, 1);
            shadedErrorBar(cols, m, e, ...
                'lineprops',{'Color',colours(path,:),'LineStyle',ls_actual{k}}, ...
                'patchSaturation',0.075);
        end

        % -------- NEW: flash times as "horizontal" segments in rotated view ----
        t1 = first_len(cond);
        t2 = first_len(cond) + second_len(cond);   % second flash time

        if path <= 2
            yseg = [0 0.5];     % left side (paths 1-2)
        else
            yseg = [0.5 1.0];   % right side (paths 3-4)
        end

        % draw at x=t1 and x=t2; colored by path; linestyle per k (1st/2nd line in subplot)
        line([t1 t1], yseg, 'Color', colours(path,:), 'LineWidth', 2, 'LineStyle', ls_flash{k});
        line([t2 t2], yseg, 'Color', colours(path,:), 'LineWidth', 2, 'LineStyle', ls_flash{k});
    end

    xlim([0 Tmax]); ylim([0 1]);
    view(-90,90);
    set(gca,'YDir','reverse');
    xlabel('Decoding (0 ↔ 1)'); ylabel('time (ms)');
    title('Paths 1 & 3 (all trials pooled)');

    % ================= BOTTOM subplot : paths 2 & 4 =======================
    subplot(2,1,2); hold on
    paths_bot = [2 4];

    for k = 1:2
        path  = paths_bot(k);
        cond  = (maze-1)*4 + path;

        data  = decoding_L_R_all{cond};
        dataS = decoding_L_R_shuffle_all{cond};

        if ~isempty(dataS)
            cols = 1:(size(dataS,2)-1);
            m = nanmean(dataS(:, cols), 1);
            e = 0.5 * nanstd(dataS(:, cols), 0, 1);
            shadedErrorBar(cols, m, e, ...
                'lineprops',{'Color','k','LineStyle',ls_shuffle{k}}, ...
                'patchSaturation',0.05);
        end

        if ~isempty(data)
            cols = 1:(size(data,2)-1);
            m = nanmean(data(:, cols), 1);
            e = 0.5 * nanstd(data(:, cols), 0, 1);
            shadedErrorBar(cols, m, e, ...
                'lineprops',{'Color',colours(path,:),'LineStyle',ls_actual{k}}, ...
                'patchSaturation',0.075);
        end

        % -------- NEW: flash times as "horizontal" segments in rotated view ----
        t1 = first_len(cond);
        t2 = first_len(cond) + second_len(cond);

        if path <= 2
            yseg = [0 0.5];
        else
            yseg = [0.5 1.0];
        end

        line([t1 t1], yseg, 'Color', colours(path,:), 'LineWidth', 2, 'LineStyle', ls_flash{k});
        line([t2 t2], yseg, 'Color', colours(path,:), 'LineWidth', 2, 'LineStyle', ls_flash{k});
    end

    xlim([0 Tmax]); ylim([0 1]);
    view(-90,90);
    set(gca,'XDir','reverse');
    set(gca,'YDir','reverse');
    xlabel('Decoding (0 ↔ 1)'); ylabel('time (ms)');
    title('Paths 2 & 4 (all trials pooled)');

    % ---------------- save & close ----------------------------------------
    fname = sprintf('%sAll_Sessions_Decoding_ST_Maze_%d', save_directory_PCA, maze);
    savefig(gcf, [fname '.fig']);
    close(gcf);
end


end



