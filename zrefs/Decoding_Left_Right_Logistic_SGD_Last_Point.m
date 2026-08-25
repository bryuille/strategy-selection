function [] = Decoding_Left_Right_Logistic_SGD_Last_Point(n_t,PROBE, MONKEY)

% Set the seed for reproducibility
seed = 45;  % Use any integer you prefer
rng(seed, 'twister');  % Set the random number generator

addpath(genpath('/om2/user/mramadan/Neurophys/Sorting/'));


data_directory= strcat('/om2/user/mramadan/Neurophys/Sorting/Analysis/', PROBE, '/', MONKEY, '/', 'Good_Trials', '/'); 

FR_directory = strcat('/om2/user/mramadan/Neurophys/Sorting/Analysis/', PROBE, '/' , MONKEY, '/Firing_Rates/');

om4_directory = strcat('/om4/group/jazlab/Mahdi/Neurophys/Sorting/Sorted_Data/',PROBE,'/',MONKEY,'/');

save_directory_PCA = strcat('/om2/user/mramadan/Neurophys/Sorting/Analysis/', PROBE, '/', MONKEY, '/Decoding_Figs/Left_Right_Flash_One_To_Flash_Three_Plus_300/Logistic_SGD_Last_Point/');



files = dir(om4_directory);
directoryNames = {files.name};
directoryNames = directoryNames(~ismember(directoryNames,{'.','..','.DS_Store'}));

sess = directoryNames{n_t};


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

nrn_train = cell(1,24);
nrn_train_full = cell(1,24);
nrn_99 = [];
nrn_test = cell(1,24);

label_train = cell(1,24);
label_train_full = cell(1,24);
label_99 = [];
label_test = cell(1,24);


path_train = cell(1,24);
path_train_full = cell(1,24);
path_99 = [];
path_test = cell(1,24);

IC_answer_train = cell(1,25);
choice_per_trial = cell(1,24);   
choice_already_built = false;  
 
 file_name = strcat(data_directory,sess,'_good_trials_concat.mat');
 display(file_name)
 All_Data = load(file_name);

 Grid_hole = All_Data.save_all_data.Grid_hole;
 Area = All_Data.save_all_data.Area;
 probe_type = All_Data.save_all_data.probe_type;
 reward = All_Data.save_all_data.reward;
 feedback =  All_Data.save_all_data.feedback_time;
 date_of_recording = All_Data.save_all_data.date_of_recording;
 AllSyncTimes = All_Data.save_all_data.all_synctimes;
 trial_fade = All_Data.save_all_data.trial_fade;
 h = All_Data.save_all_data.h;
 h1 = All_Data.save_all_data.h1;
 h2 = All_Data.save_all_data.h2;
 h3 = All_Data.save_all_data.h3;
 h4 = All_Data.save_all_data.h4;
 h5 = All_Data.save_all_data.h5;
 h6 = All_Data.save_all_data.h6;
 vel = All_Data.save_all_data.vel;
 LR = All_Data.save_all_data.LR;
 LR2 = All_Data.save_all_data.LR2;
 trial_answer1 = All_Data.save_all_data.trial_answer1;
 trial_answer2 = All_Data.save_all_data.trial_answer2;
 trial_answer3 = All_Data.save_all_data.trial_answer3;
 trial_answer4 = All_Data.save_all_data.trial_answer4;
 trial_time_to_fix = All_Data.save_all_data.trial_time_to_fix;
 which_sync = All_Data.save_all_data.which_sync;
 geo_present = All_Data.save_all_data.geo_present;
 fixation_cue_present = All_Data.save_all_data.fixation_cue_present;
 fix_start = All_Data.save_all_data.fix_start;
 flash_one = All_Data.save_all_data.flash_one;
 flash_two = All_Data.save_all_data.flash_two;
 flash_three = All_Data.save_all_data.flash_three;
 fixation_off = All_Data.save_all_data.fixation_off;
 saccade_init = All_Data.save_all_data.saccade_init;
 answer_time= All_Data.save_all_data.answer_time;
 trial_end = All_Data.save_all_data.trial_end;
 sync_params = All_Data.save_all_data.sync_params;
 spikes =  All_Data.save_all_data.spikes;
 cids = All_Data.save_all_data.cids;
 nrns_all = All_Data.save_all_data.nrns;
 labels = All_Data.save_all_data.labels;
 spikes_all = All_Data.save_all_data.spikes_all;
 trial_indices_all = All_Data.save_all_data.trial_indices_all;
 geo_type = All_Data.save_all_data.geo_type;
 path_type = All_Data.save_all_data.path_type;
 one_hier = All_Data.save_all_data.one_hier;
 two_hier = All_Data.save_all_data.two_hier;
 two_hier_left = All_Data.save_all_data.two_hier_left;
 two_hier_right = All_Data.save_all_data.two_hier_right;
 two_hier_correct= All_Data.save_all_data.two_hier_correct;
 two_hier_alt = All_Data.save_all_data.two_hier_alt;
 one = All_Data.save_all_data.one;
 two = All_Data.save_all_data.two;
 rand_geo = All_Data.save_all_data.rand_geo;

FR_WH_LOAD = load(strcat(FR_directory, sess, '_Whole_Trial_FR_Causal.mat'));
FR_WH = FR_WH_LOAD.smooth_session;

%%


neurons = cell(length(unique(cids)),1);

[mode_trial, model_idx]= mode(nrns_all);

for nrns = 1:length(unique(cids))
     
     trial_list = trial_indices_all(nrns_all == nrns);
     path_type_t = path_type(nrns_all == nrns);  
     trial_fade_t = trial_fade(nrns_all == nrns);   
     select_nrn_trials = find( (path_type_t ~= -99) & (trial_fade_t == 0));
     
     
     trial_list(select_nrn_trials);
     neurons{nrns,1} = trial_list(select_nrn_trials);

end


all_trials = unique(cell2mat(neurons));

nrn_bolean = [];
start_trials = [];
end_trials = [];

for yi = 1:length(all_trials)-1
    yi;
    for iy = 0:length(all_trials)-yi
         
         
         temp_nrn = [];

         for nnn = 1:numel(neurons)        
                temp_nrn = horzcat(temp_nrn, (length(intersect(all_trials(yi:length(all_trials)-iy) , neurons{nnn})) == length(all_trials(yi:length(all_trials)-iy) ) )  );
         end

       
        nrn_bolean = vertcat(nrn_bolean, sum(temp_nrn) );
        start_trials = vertcat(start_trials, all_trials(yi));
        end_trials = vertcat(end_trials,all_trials(length(all_trials)-iy));


    end

end

percent_neuron = nrn_bolean./numel(neurons);
percent_trial = (end_trials-start_trials)./(end_trials(1)-start_trials(1));
dif_percent_neuron_trial = abs( percent_neuron - percent_trial) ; 

[best_value,best_index] = max( percent_neuron + percent_trial - dif_percent_neuron_trial); 

choose_trials =  (start_trials(best_index) :  end_trials(best_index));


min_value = start_trials(best_index);
max_value = end_trials(best_index);


keep_track_neurons = [];
%%

keep_track = 0;
for nrns = 1:length(unique(cids))

    if length(intersect(choose_trials, trial_indices_all(nrns_all == nrns))) == length(choose_trials)  
            
             keep_track_neurons = vertcat(keep_track_neurons, nrns);
             keep_track = keep_track +1;

             good_trials = spikes_all((nrns_all == nrns),:);
             geo_present_t = geo_present(nrns_all== nrns);
             fixation_cue_present_t = fixation_cue_present(nrns_all == nrns);
             fix_start_t = fix_start(nrns_all == nrns);
             flash_one_t = flash_one(nrns_all == nrns);
             flash_two_t = flash_two(nrns_all == nrns);
             flash_three_t = flash_three(nrns_all == nrns);
             fixation_off_t = fixation_off(nrns_all == nrns);
             saccade_init_t = saccade_init(nrns_all == nrns);
             answer_time_t = answer_time(nrns_all == nrns);
             trial_end_t = trial_end(nrns_all == nrns);
             trial_answer1_t = trial_answer1(nrns_all == nrns);
             trial_answer2_t = trial_answer2(nrns_all == nrns);
             trial_answer3_t = trial_answer3(nrns_all == nrns);
             trial_answer4_t = trial_answer4(nrns_all == nrns);
             geo_type_t = geo_type(nrns_all == nrns);
             path_type_t = path_type(nrns_all == nrns);             
             one_hier_t = one_hier(nrns_all == nrns);
             two_hier_t = two_hier(nrns_all == nrns);
             two_hier_left_t = two_hier_left(nrns_all == nrns);
             two_hier_right_t = two_hier_right(nrns_all == nrns);
             two_hier_correct_t= two_hier_correct(nrns_all == nrns);
             two_hier_alt_t = two_hier_alt(nrns_all == nrns);
             one_t = one(nrns_all == nrns);
             two_t = two(nrns_all == nrns);
             rand_geo_t = rand_geo(nrns_all == nrns);
             cids_t = cids(nrns_all == nrns);
             trial_fade_t = trial_fade(nrns_all == nrns);
             trial_list = trial_indices_all(nrns_all == nrns);
             LR_t = LR(nrns_all == nrns);
             LR2_t = LR2(nrns_all == nrns);

             smooth_all_t= FR_WH(nrns_all == nrns,:);

             % Z-score
             zscor_xnan = @(x,all_data) bsxfun(@rdivide, bsxfun(@minus, x, nanmean(all_data(:)) ), nanstd(all_data(:)) );      
             smooth_all = zscor_xnan(smooth_all_t,smooth_all_t);
            
             % Find trial with maximum length
             max_length = max(arrayfun(@(trl) floor(flash_three_t(trl)*1000 - geo_present_t(trl)*1000) + 300 - (floor(flash_one_t(trl)*1000 - geo_present_t(trl)*1000) + 1) + 1, 1:size(smooth_all,1)));

            % Extract data and pad with NaN for shorter trials            
            extracted_data = arrayfun(@(trl) padarray(smooth_all(trl, (floor(flash_one_t(trl)*1000 - geo_present_t(trl)*1000) + 1) : floor(flash_three_t(trl)*1000 - geo_present_t(trl)*1000) + 300), ...
            [0, max_length - (floor(flash_three_t(trl)*1000 - geo_present_t(trl)*1000) + 300 - (floor(flash_one_t(trl)*1000 - geo_present_t(trl)*1000) + 1) + 1)], NaN, 'post'), ...
            1:size(smooth_all,1), 'UniformOutput', false);

            smooth_all = vertcat(extracted_data{:});  % Convert cell array into matrix



%              % find indicies of occluded trials to test
%              [ v, indx_use, indx_dont_use]  = intersect(trial_indices_all(nrns_all == nrns), choose_trials);
%              endpoints = sum(~isnan(smooth_all(indx_use,:)),2);
            

            nrn_99_temp = [];
            label_99_temp =[];
            path_99_temp = [];

            can_use = 1;

            for cond = 1:24     
                visible_indices = find( (path_type_t == cond) & (trial_fade_t == 0));   
                indx_use =  trial_list(visible_indices);
                within_range_indices = (indx_use >= min_value) & (indx_use <= max_value);
                values_within_range = visible_indices (within_range_indices);

                if length(values_within_range)< 2
                    can_use = 0;
                end
            end
            
            if can_use == 1
                for cond = 1:24
                 
                    visible_indices = find( (path_type_t == cond) & (trial_fade_t == 0));   
                    indx_use =  trial_list(visible_indices);
                    within_range_indices = (indx_use >= min_value) & (indx_use <= max_value);
                    values_within_range = visible_indices (within_range_indices);
                    
                    nrn_train_temp = [];
                    nrn_train_temp_full = [];
                    endpoints = sum(~isnan(smooth_all(values_within_range,:)),2);
                    for ends = 1:length(values_within_range)

%                        nrn_train_temp = vertcat(nrn_train_temp, [smooth_all(values_within_range(ends), round((fixation_off_t(values_within_range(ends)) - flash_one_t(values_within_range(ends)))*1000) - 200 : round((fixation_off_t(values_within_range(ends)) - flash_one_t(values_within_range(ends)))*1000)) ] );        
                         nrn_train_temp = vertcat(nrn_train_temp, [smooth_all(values_within_range(ends),  endpoints(ends) - 300 : endpoints(ends) ) ] );
%                        nrn_train_temp = vertcat(nrn_train_temp, [smooth_all(values_within_range(ends), round(trial_time_to_fix_t(values_within_range(ends)) + (flash_three_t(values_within_range(ends)) - flash_one_t(values_within_range(ends)))*1000) - 300 : round(trial_time_to_fix_t(values_within_range(ends)) + (flash_three_t(values_within_range(ends)) - flash_one_t(values_within_range(ends)))*1000) ) ] );        
%                        nrn_train_temp = vertcat(nrn_train_temp, [smooth_all(values_within_range(ends), round((flash_three_t(values_within_range(ends)) - flash_one_t(values_within_range(ends)))*1000):round((flash_three_t(values_within_range(ends)) - flash_one_t(values_within_range(ends)))*1000)+300) ] );

                       nrn_train_temp_full = vertcat( nrn_train_temp_full, [smooth_all(values_within_range(ends), 1:stop_len(cond)) ] );


                    end
                    
                    IC_answer_train{cond} = cat(3, IC_answer_train{cond}, (trial_answer3_t(values_within_range)| trial_answer4_t(values_within_range)));
           
                    nrn_train{cond} = cat(3, nrn_train{cond}, nrn_train_temp );
                    nrn_train_full{cond} = cat(3, nrn_train_full{cond},  nrn_train_temp_full );


                     % left = 0, right = 1   
                     L_R_train_test = [];
        
                      for l_r = 1:length(values_within_range)
                          
                          index = values_within_range(l_r);
                         
                          if LR_t(index) == -1
                                
                            if (trial_answer1_t(index) == 1) | (trial_answer2_t(index) == 1)
                                
                                 L_R_train_test = vertcat( L_R_train_test , 0);
                            
                            elseif (trial_answer3_t(index) == 1) | (trial_answer4_t(index) == 1)
        
                                 L_R_train_test = vertcat( L_R_train_test , 1);
                            end
        
                          elseif LR_t(index) == 1
        
                             if (trial_answer1_t(index) == 1) | (trial_answer2_t(index) == 1)
                                
                                 L_R_train_test = vertcat( L_R_train_test , 1);
                            
                            elseif (trial_answer3_t(index) == 1) | (trial_answer4_t(index) == 1)
        
                                 L_R_train_test = vertcat( L_R_train_test , 0);
                            end
        
        
                          end
        
                      end
                    
                    if ~choice_already_built
                        % ---------------- compute 4-way final choice per trial ----------------
                        % 1 = left-up, 2 = left-down, 3 = right-up, 4 = right-down
                        final_choice = nan(length(values_within_range), 1);
                        
                        for l_r = 1:length(values_within_range)
                            index = values_within_range(l_r);
                        
                            if trial_answer1_t(index) == 1 % correct exit
                                if LR_t(index) == -1 && LR2_t(index) ==  1
                                    final_choice(l_r) = 1; % LU
                                elseif LR_t(index) == -1 && LR2_t(index) == -1
                                    final_choice(l_r) = 2; % LD
                                elseif LR_t(index) ==  1 && LR2_t(index) ==  1
                                    final_choice(l_r) = 3; % RU
                                elseif LR_t(index) ==  1 && LR2_t(index) == -1
                                    final_choice(l_r) = 4; % RD
                                end
                        
                            elseif trial_answer2_t(index) == 1 % correct L/R but wrong U/D (flip U/D)
                                if LR_t(index) == -1 && LR2_t(index) ==  1
                                    final_choice(l_r) = 2; % chose LD
                                elseif LR_t(index) == -1 && LR2_t(index) == -1
                                    final_choice(l_r) = 1; % chose LU
                                elseif LR_t(index) ==  1 && LR2_t(index) ==  1
                                    final_choice(l_r) = 4; % chose RD
                                elseif LR_t(index) ==  1 && LR2_t(index) == -1
                                    final_choice(l_r) = 3; % chose RU
                                end
                        
                            elseif trial_answer3_t(index) == 1 % wrong L/R, correct U/D (flip L/R)
                                if LR_t(index) == -1 && LR2_t(index) ==  1
                                    final_choice(l_r) = 3; % chose RU
                                elseif LR_t(index) == -1 && LR2_t(index) == -1
                                    final_choice(l_r) = 4; % chose RD
                                elseif LR_t(index) ==  1 && LR2_t(index) ==  1
                                    final_choice(l_r) = 1; % chose LU
                                elseif LR_t(index) ==  1 && LR2_t(index) == -1
                                    final_choice(l_r) = 2; % chose LD
                                end
                        
                            elseif trial_answer4_t(index) == 1 % wrong L/R and wrong U/D (flip both)
                                if LR_t(index) == -1 && LR2_t(index) ==  1
                                    final_choice(l_r) = 4; % chose RD
                                elseif LR_t(index) == -1 && LR2_t(index) == -1
                                    final_choice(l_r) = 3; % chose RU
                                elseif LR_t(index) ==  1 && LR2_t(index) ==  1
                                    final_choice(l_r) = 2; % chose LD
                                elseif LR_t(index) ==  1 && LR2_t(index) == -1
                                    final_choice(l_r) = 1; % chose LU
                                end
                            end
                        end
                    
                        % Append (keeps trial order identical to everything else)
                        choice_per_trial{cond} = [choice_per_trial{cond}; final_choice];
                        % -------------------------------------------------------------------------
                    end
                
                     
                     label_train{cond} = cat(3,label_train{cond},L_R_train_test.*ones(1,300+1));
                      path_train{cond} = cat(3,path_train{cond},path_type_t(values_within_range).*ones(1,300+1));

                      label_train_full{cond} = cat(3,label_train_full{cond},L_R_train_test.*ones(1,stop_len(cond)));
                      path_train_full{cond} = cat(3,path_train_full{cond},path_type_t(values_within_range).*ones(1,stop_len(cond)));
            
            

                end



            for cond = -99
                 
                    visible_indices = find( (path_type_t == cond) & (trial_fade_t == 0));   
                    indx_use =  trial_list(visible_indices);
                    within_range_indices = (indx_use >= min_value) & (indx_use <= max_value);
                    values_within_range = visible_indices (within_range_indices);

                    endpoints = sum(~isnan(smooth_all(values_within_range,:)),2);
                    for ends = 1:length(values_within_range)
                        nrn_99_temp= vertcat(nrn_99_temp, nanmean(smooth_all(values_within_range(ends), endpoints(ends) - 100 : endpoints(ends) )) );
                    end


                       % left = 0, right = 1   
                     L_R_train_test = [];
        
                      for l_r = 1:length(values_within_range)
                          
                          index = values_within_range(l_r);
                         
                          if LR_t(index) == -1
                                
                            if (trial_answer1_t(index) == 1) | (trial_answer2_t(index) == 1)
                                
                                 L_R_train_test = vertcat( L_R_train_test , 0);
                            
                            elseif (trial_answer3_t(index) == 1) | (trial_answer4_t(index) == 1)
        
                                 L_R_train_test = vertcat( L_R_train_test , 1);
                            end
        
                          elseif LR_t(index) == 1
        
                             if (trial_answer1_t(index) == 1) | (trial_answer2_t(index) == 1)
                                
                                 L_R_train_test = vertcat( L_R_train_test , 1);
                            
                            elseif (trial_answer3_t(index) == 1) | (trial_answer4_t(index) == 1)
        
                                 L_R_train_test = vertcat( L_R_train_test , 0);
                            end
        
        
                          end
        
                      end
                
                   
                      label_99_temp = vertcat(label_99_temp,L_R_train_test);
                      path_99_temp =  vertcat( path_99_temp,path_type_t(values_within_range));


            end
            
             choice_already_built = true;
             
            end
            
            nrn_99 = horzcat(nrn_99, nrn_99_temp);
            label_99 = horzcat(label_99, label_99_temp);
            path_99 = horzcat(path_99, path_99_temp);

    end

end


label_99 = mode(label_99 ,2);

for i = 1:numel(label_train)
    label_train{i} = mode(label_train{i},3);
end

for i = 1:numel(label_train_full)
    label_train_full{i} = mode(label_train_full{i},3);
end

for i = 1:numel(label_test)
    label_test{i} = mode(label_test{i},2);
end

path_99 = mode(path_99 ,2);

for i = 1:numel(path_train)
    path_train{i} = mode(path_train{i},3);
end

for i = 1:numel(path_test)
    path_test{i} = mode(path_test{i},2);
end


flatCells = cellfun(@(x) reshape(permute(x, [2, 1, 3]), [], size(x,3)), nrn_train(1:24), 'UniformOutput', false);
nrn_train_cat = vertcat(flatCells{:});

flatCells = cellfun(@(x) reshape(x', [], size(x,3)), label_train(1:24), 'UniformOutput', false);
nrn_label_cat = vertcat(flatCells{:});

flatCells = cellfun(@(x) reshape(x', [], size(x,3)), path_train(1:24), 'UniformOutput', false);
nrn_path_cat = vertcat(flatCells{:});

trial_count_labels = cellfun(@(x) x(:,1) ,  label_train(1:24), 'UniformOutput', false);
trial_count_labels_cat = vertcat(trial_count_labels{:});

size(nrn_train{1},3)

%% FIND OPTIMAL LAMDA FOR DECODING


Lambda = logspace(-6,-0.5,11);

CVMdl = fitclinear(nrn_train_cat,nrn_label_cat,'KFold',10,...
    'Learner','logistic','Solver','sgd','Regularization','ridge','OptimizeLearnRate',true, ...
    'Lambda',Lambda,'Prior', 'uniform');
ce = kfoldLoss(CVMdl);


Mdl = fitclinear(nrn_train_cat,nrn_label_cat,...
    'Learner','logistic','Solver','sgd','Regularization','ridge','OptimizeLearnRate',true, ...
    'Lambda',Lambda,'Prior', 'uniform');
numNZCoeff = sum(abs(Mdl.Beta));

figure;
[h,hL1,hL2] = plotyy(log10(Lambda),log10(ce),...
    log10(Lambda),log10(numNZCoeff)); 
hL1.Marker = 'o';
hL2.Marker = 'o';
ylabel(h(1),'log_{10} classification error')
ylabel(h(2),'log_{10} coefficient sum')
xlabel('log_{10} Lambda')
title('Test-Sample Statistics')
hold off


% pick lambda that balances classification error with magnitude of Betas
[best_lamdba_value, best_lamda_index] = min([normalize(log10(ce), 'range') + normalize(log10(numNZCoeff), 'range') + abs(normalize(log10(ce), 'range') - normalize(log10(numNZCoeff), 'range'))]);
best_lamda = Lambda(best_lamda_index);
saveas( gcf , strcat(save_directory_PCA,'Optimal_Lambda_',sess) );

%% FIND OPTIMAL TIMEPOINT FOR DECODING

all_bootstrap_mean = [];
all_bootstrap_std = [];

all_shuffled_bootstrap_mean = [];
all_shuffled_bootstrap_std = [];

bootstrap_mean = [];
shuffled_bootstrap_mean = [];

for iter = 1:10
    
    test_size = 0.5;    
    cv = cvpartition(size(trial_count_labels_cat,1),'HoldOut',test_size);
    idx = cv.test;

    originalSizes = cellfun(@(x) x(1) , cellfun(@size, label_train(1:24), 'UniformOutput', false));
    % Reshape the concatenated array back into the original arrays
    cellArrayReshaped = cell(size(label_train(1:24)));
    startIdx = 1;
    for i = 1:numel(label_train(1:24))
        cellSize = [originalSizes(i),1];  
        numElements = prod(cellSize);
        cellArrayReshaped{i} = reshape(idx(startIdx:startIdx+numElements-1), cellSize);
        startIdx = startIdx + numElements;
    end
    
    nrn_train_iter  = cellfun(@(x, idx) x(~idx, :, :), nrn_train(1:24), cellArrayReshaped, 'UniformOutput', false);
    nrn_label_iter = cellfun(@(x, idx) x(~idx, :), label_train(1:24), cellArrayReshaped, 'UniformOutput', false);
    
    nrn_train_iter_test  = cellfun(@(x, idx) x(idx, :, :), nrn_train(1:24), cellArrayReshaped, 'UniformOutput', false);
    nrn_label_iter_test = cellfun(@(x, idx) x(idx, :), label_train(1:24), cellArrayReshaped, 'UniformOutput', false);

    flatCells = cellfun(@(x) reshape(permute(x, [2, 1, 3]), [], size(x,3)), nrn_train_iter, 'UniformOutput', false);
    nrn_train_iter_cat = vertcat(flatCells{:});

    flatCells = cellfun(@(x) reshape(x', [], size(x,3)),  nrn_label_iter, 'UniformOutput', false);
    nrn_label_iter_cat = vertcat(flatCells{:});

    flatCells = cellfun(@(x) reshape(permute(x, [2, 1, 3]), [], size(x,3)), nrn_train_iter_test, 'UniformOutput', false);
     nrn_train_iter_test_cat = vertcat(flatCells{:});

    flatCells = cellfun(@(x) reshape(x', [], size(x,3)),  nrn_label_iter_test, 'UniformOutput', false);
    nrn_label_iter_test_cat = vertcat(flatCells{:});

    Model = fitclinear(nrn_train_iter_cat,nrn_label_iter_cat,...
    'Learner','logistic','Solver','SGD','Regularization','ridge','OptimizeLearnRate',true,...
    'Lambda',best_lamda,'Prior', 'uniform');

    val_pred = predict(Model, nrn_train_iter_test_cat);
    
   bootstrap_mean = vertcat(bootstrap_mean, mean((val_pred >= 0.5) == nrn_label_iter_test_cat)*100);
   

    sub_label_train = nrn_label_iter_cat;

    Model = fitclinear(nrn_train_iter_cat,sub_label_train(randperm(size(sub_label_train, 1))),...
    'Learner','logistic','Solver','SGD','Regularization','ridge','OptimizeLearnRate',true,...
    'Lambda',best_lamda,'Prior', 'uniform');

    val_pred = predict(Model,  nrn_train_iter_test_cat);
    
  shuffled_bootstrap_mean = vertcat(shuffled_bootstrap_mean, mean((val_pred >= 0.5) == nrn_label_iter_test_cat)*100);
 

end

all_bootstrap_mean = vertcat(all_bootstrap_mean,mean(bootstrap_mean));
all_bootstrap_std = vertcat(all_bootstrap_std ,std(bootstrap_mean));

all_shuffled_bootstrap_mean = vertcat(all_shuffled_bootstrap_mean,mean(shuffled_bootstrap_mean));
all_shuffled_bootstrap_std = vertcat(all_shuffled_bootstrap_std,std(shuffled_bootstrap_mean));


  figure()
  scatter(1,all_bootstrap_mean, 'r')
  hold on
  errorbar(1,all_bootstrap_mean, all_bootstrap_std, 'r')
  ylim([0 100])

  hold on 

  scatter(1,all_shuffled_bootstrap_mean, 'k')
  hold on
  errorbar(1,all_shuffled_bootstrap_mean,all_shuffled_bootstrap_std , 'k')
  ylim([0 100])

 saveas( gcf , strcat(save_directory_PCA,'Optimal_Timepoint_',sess) );

 [ best_decoding_t_value, best_decoding_t_index ]= max(all_bootstrap_mean);
 save(strcat(save_directory_PCA,'_decoding_perf_within_', sess, '.mat'), 'best_decoding_t_value','-v7.3');
 

%% LOOK AT PCA EMBEDDING


nrn_av = nrn_train_cat  - nanmean(nrn_train_cat,1);

[coeff,score,latent,tsquared,explained,mu] = pca(nrn_av);
figure()
plot(1:length(explained), cumsum(explained),'-k','LineWidth',2)
title('scree plot')
ylabel('percent var explained')
xlabel('PC')
ylim([0 100])


scatter3(score(nrn_label_cat == 1,1),score(nrn_label_cat == 1,2),score(nrn_label_cat == 1,3), 'r')
hold on
scatter3(score(nrn_label_cat == 0,1),score(nrn_label_cat == 0,2),score(nrn_label_cat == 0,3), 'k')

saveas( gcf , strcat(save_directory_PCA,'PCA_L_R_',sess) );

%% Train logistic classifier


model_pred_accuracy = cell(1,24);
model_pred_L_R = cell(1,24);
model_pred_L_R_trials = cell(1,24);
model_pred_L_R_trials_sigmoid = cell(1,24);
model_pred_L_R_trials_raw = cell(1,24);

model_pred_accuracy_shuffle = cell(1,24);
model_pred_L_R_shuffle = cell(1,24);
model_pred_L_R_trials_shuffle = cell(1,24);
model_pred_L_R_trials_shuffle_sigmoid= cell(1,24);
model_pred_L_R_trials_shuffle_raw= cell(1,24);

sig_mahdi = @(vec) 1./(1+exp(- vec));

for iter = 1:200


test_size = 0.5;    
cv = cvpartition(size(trial_count_labels_cat,1),'HoldOut',test_size);
idx = cv.test;

% ensure equal L/R
minLabelCount = min([length(trial_count_labels_cat((trial_count_labels_cat == 0) & ~idx)), length(trial_count_labels_cat((trial_count_labels_cat == 1) & ~idx))]);

indices_0 = find((trial_count_labels_cat == 0) & ~idx);
indices_1 = find((trial_count_labels_cat == 1) & ~idx);

[sampledData_0, idx_0 ] = datasample(trial_count_labels_cat((trial_count_labels_cat == 0) & ~idx), minLabelCount, 1, 'Replace', false);
[sampledData_1, idx_1 ] = datasample(trial_count_labels_cat((trial_count_labels_cat == 1) & ~idx), minLabelCount, 1, 'Replace', false);  

unsampledIndices_0 = setdiff(1:size(trial_count_labels_cat((trial_count_labels_cat == 0) & ~idx)), idx_0);
unsampledIndices_1 = setdiff(1:size(trial_count_labels_cat((trial_count_labels_cat == 1) & ~idx)), idx_1);

idx(indices_0(unsampledIndices_0)) = 1;
idx(indices_1(unsampledIndices_1)) = 1;

originalSizes = cellfun(@(x) x(1) , cellfun(@size, label_train(1:24), 'UniformOutput', false));
% Reshape the concatenated array back into the original arrays
cellArrayReshaped = cell(size(label_train(1:24)));
startIdx = 1;
for i = 1:numel(label_train(1:24))
    cellSize = [originalSizes(i),1];  
    numElements = prod(cellSize);
    cellArrayReshaped{i} = reshape(idx(startIdx:startIdx+numElements-1), cellSize);
    startIdx = startIdx + numElements;
end

nrn_train_iter  = cellfun(@(x, idx) x(~idx, :, :), nrn_train(1:24), cellArrayReshaped, 'UniformOutput', false);
nrn_label_iter = cellfun(@(x, idx) x(~idx, :), label_train(1:24), cellArrayReshaped, 'UniformOutput', false);

nrn_train_iter_test  = cellfun(@(x, idx) x(idx, :, :), nrn_train_full(1:24), cellArrayReshaped, 'UniformOutput', false);
nrn_label_iter_test = cellfun(@(x, idx) x(idx, :), label_train_full(1:24), cellArrayReshaped, 'UniformOutput', false);

flatCells = cellfun(@(x) reshape(permute(x, [2, 1, 3]), [], size(x,3)), nrn_train_iter, 'UniformOutput', false);
nrn_train_iter_cat = vertcat(flatCells{:});

flatCells = cellfun(@(x) reshape(x', [], size(x,3)),  nrn_label_iter, 'UniformOutput', false);
nrn_label_iter_cat = vertcat(flatCells{:});


sampledData_all = nrn_train_iter_cat;
sampled_labels_all = nrn_label_iter_cat;

% Generate a permutation of indices
perm = randperm(size(sampledData_all,1))';

 Model = fitclinear(sampledData_all(perm,:),sampled_labels_all(perm),...
    'Learner','logistic','Solver','sgd','Regularization','ridge','OptimizeLearnRate',true, ...
    'Lambda',best_lamda, 'Prior', 'uniform');


c = 0;
for conds = 1:24

 c = c+1;
 accur_test = [];
 L_R_trial_test = nan(size(nrn_train_full{conds},1), size(nrn_train_full{conds},2));
 L_R_trial_test_raw = nan(size(nrn_train_full{conds},1), size(nrn_train_full{conds},2));
 L_R_trial_test_sigmoid = nan(size(nrn_train_full{conds},1), size(nrn_train_full{conds},2));
 idx_test = cellArrayReshaped{c};

 nrn_cond_test = nrn_train_iter_test{c};
 label_cond_test = nrn_label_iter_test{c};

for prd = 1 : size(nrn_cond_test,2)

% Predict decisions
Ap = permute(nrn_cond_test(:,prd,:), [1 3 2]);
B = reshape(Ap, size(Ap, 1), size(Ap, 2));
predicted_labels = predict(Model, B);

accur_test = horzcat(accur_test, predicted_labels);

L_R_trial_test(idx_test,prd) = predict(Model, B);
L_R_trial_test_raw(idx_test,prd) = B*Model.Beta + Model.Bias;
L_R_trial_test_sigmoid(idx_test,prd) = sig_mahdi(B*Model.Beta + Model.Bias);

end

model_pred_accuracy{conds} = vertcat(model_pred_accuracy{conds}, mean((accur_test >=0.5) ==  label_cond_test,1));
model_pred_L_R{conds} = vertcat(model_pred_L_R{conds}, mean(accur_test,1));

model_pred_L_R_trials{conds} = cat(3, model_pred_L_R_trials{conds}, L_R_trial_test);
model_pred_L_R_trials_raw{conds} = cat(3, model_pred_L_R_trials_raw{conds}, L_R_trial_test_raw );
model_pred_L_R_trials_sigmoid{conds} = cat(3, model_pred_L_R_trials_sigmoid{conds}, L_R_trial_test_sigmoid );

end



sub_label_train = sampled_labels_all(perm);

 Model = fitclinear(sampledData_all(perm,:),[sub_label_train(randperm(size(sub_label_train, 1)))],...
    'Learner','logistic','Solver','sgd','Regularization','ridge','OptimizeLearnRate',true,...
    'Lambda',best_lamda, 'Prior', 'uniform');

  c = 0;
for conds = 1:24
 c = c+1;
 accur_test = [];
 L_R_trial_test = nan(size(nrn_train_full{conds},1), size(nrn_train_full{conds},2));
 L_R_trial_test_raw = nan(size(nrn_train_full{conds},1), size(nrn_train_full{conds},2));
 L_R_trial_test_sigmoid = nan(size(nrn_train_full{conds},1), size(nrn_train_full{conds},2));
 idx_test = cellArrayReshaped{c};

 nrn_cond_test = nrn_train_iter_test{c};
 label_cond_test = nrn_label_iter_test{c};

for prd = 1 : size(nrn_cond_test,2)

% Predict decisions
Ap = permute(nrn_cond_test(:,prd,:), [1 3 2]);
B = reshape(Ap, size(Ap, 1), size(Ap, 2));
predicted_labels = predict(Model, B);

accur_test = horzcat(accur_test, predicted_labels);

L_R_trial_test(idx_test,prd) = predict(Model, B);
L_R_trial_test_raw(idx_test,prd) = B*Model.Beta + Model.Bias;
L_R_trial_test_sigmoid(idx_test,prd) = sig_mahdi(B*Model.Beta + Model.Bias);
end

model_pred_accuracy_shuffle{conds} = vertcat(model_pred_accuracy_shuffle{conds}, mean((accur_test >=0.5) ==  label_cond_test,1));
model_pred_L_R_shuffle{conds} = vertcat(model_pred_L_R_shuffle{conds}, mean(accur_test,1));
model_pred_L_R_trials_shuffle{conds} = cat(3, model_pred_L_R_trials_shuffle{conds}, L_R_trial_test);
model_pred_L_R_trials_shuffle_raw{conds} = cat(3, model_pred_L_R_trials_shuffle_raw{conds}, L_R_trial_test_raw);
model_pred_L_R_trials_shuffle_sigmoid{conds} = cat(3, model_pred_L_R_trials_shuffle_sigmoid{conds}, L_R_trial_test_sigmoid);

end



end

% plot condition average
for conds = 1:length(stop_len)

    figure()
    subplot(1,2,1)
    shadedErrorBar(1:size(model_pred_accuracy{conds},2),model_pred_accuracy{conds}, {@nanmean,@nanstd}, 'lineprops', '-r','patchSaturation',0.075)
    hold on 
    shadedErrorBar(1:size(model_pred_accuracy_shuffle{conds},2),model_pred_accuracy_shuffle{conds}, {@nanmean,@nanstd}, 'lineprops', '-k','patchSaturation',0.05)
    xline([first_len(conds)],'Color','r','LineWidth',3); 
    xline([first_len(conds) + second_len(conds)],'Color','r','LineWidth',3); 
    xlabel('time')
    ylabel('accuracy')
    ylim ( [ 0 1])
    
    subplot(1,2,2)
    shadedErrorBar(1:size(model_pred_L_R{conds},2),model_pred_L_R{conds}, {@nanmean,@nanstd}, 'lineprops', '-r','patchSaturation',0.075)
    hold on 
    shadedErrorBar(1:size(model_pred_L_R_shuffle{conds},2),model_pred_L_R_shuffle{conds}, {@nanmean,@nanstd}, 'lineprops', '-k','patchSaturation',0.05)
 
    hold on 
    xline([first_len(conds)],'Color','r','LineWidth',3); 
    xline([first_len(conds) + second_len(conds)],'Color','r','LineWidth',3); 
    xlabel('time')
    ylabel('Left vs. Right ( -1 vs 1)')
    
    ylim ( [ 0 1])

     saveas( gcf , strcat(save_directory_PCA,'Condition_Average_Decoding_',sess, '_', num2str(conds) ) );
end


%% Single Trials Shuffle Trials

grad = [linspace(pin(1), dr(1), 4)', linspace(pin(2), dr(2), 4)',linspace(pin(3), dr(3), 4)'];
figure()
keep_track = 0;
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials_shuffle{conds},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{conds}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials_shuffle{conds}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials_shuffle{conds},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');

            ylim([-1 1])

        elseif IC_answer_train{conds}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials_shuffle{conds}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials_shuffle{conds},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
            ylim([-1 1])

        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end

saveas( gcf , strcat(save_directory_PCA,'Single_Trials_Shuffle_',sess, '_', num2str(1) ) );

grad = [linspace(pin(1), dr(1), 4)', linspace(pin(2), dr(2), 4)',linspace(pin(3), dr(3), 4)'];
figure()
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials_shuffle{keep_track},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{keep_track}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials_shuffle{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials_shuffle{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
            ylim([-1 1])

        elseif IC_answer_train{keep_track}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials_shuffle{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials_shuffle{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
            ylim([-1 1])
        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end

saveas( gcf , strcat(save_directory_PCA,'Single_Trials_Shuffle_',sess, '_', num2str(2) ) );

grad = [linspace(cy(1), dbl(1), 4)', linspace(cy(2), dbl(2), 4)',linspace(cy(3), dbl(3), 4)'];
figure()
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials_shuffle{keep_track},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{keep_track}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials_shuffle{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials_shuffle{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
            ylim([-1 1])

        elseif IC_answer_train{keep_track}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials_shuffle{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials_shuffle{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
            ylim([-1 1])

        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end

saveas( gcf , strcat(save_directory_PCA,'Single_Trials_Shuffle_',sess, '_', num2str(3) ) );

grad = [linspace(cy(1), dbl(1), 4)', linspace(cy(2), dbl(2), 4)',linspace(cy(3), dbl(3), 4)'];
figure()
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials_shuffle{keep_track},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{keep_track}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials_shuffle{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials_shuffle{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
            ylim([-1 1])

        elseif IC_answer_train{keep_track}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials_shuffle{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials_shuffle{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
            ylim([-1 1])
        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end

saveas( gcf , strcat(save_directory_PCA,'Single_Trials_Shuffle_',sess, '_', num2str(4) ) );

grad = [linspace(gld(1), dor(1), 4)', linspace(gld(2), dor(2), 4)',linspace(gld(3), dor(3), 4)'];
figure()
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials_shuffle{keep_track},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{keep_track}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials_shuffle{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials_shuffle{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
            ylim([-1 1])

        elseif IC_answer_train{keep_track}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials_shuffle{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials_shuffle{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
            ylim([-1 1])

        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end

saveas( gcf , strcat(save_directory_PCA,'Single_Trials_Shuffle_',sess, '_', num2str(5) ) );

grad = [linspace(gld(1), dor(1), 4)', linspace(gld(2), dor(2), 4)',linspace(gld(3), dor(3), 4)'];
figure()
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials_shuffle{keep_track},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{keep_track}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials_shuffle{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials_shuffle{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
            ylim([-1 1])

        elseif IC_answer_train{keep_track}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials_shuffle{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials_shuffle{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
            ylim([-1 1])
        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end

saveas( gcf , strcat(save_directory_PCA,'Single_Trials_Shuffle_',sess, '_', num2str(6) ) );

%% Single Trials

grad = [linspace(pin(1), dr(1), 4)', linspace(pin(2), dr(2), 4)',linspace(pin(3), dr(3), 4)'];
figure()
keep_track = 0;
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials{conds},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{conds}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials{conds}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{conds},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
            
        elseif IC_answer_train{conds}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials{conds}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{conds},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end

saveas( gcf , strcat(save_directory_PCA,'Single_Trials_',sess, '_', num2str(1) ) );


grad = [linspace(pin(1), dr(1), 4)', linspace(pin(2), dr(2), 4)',linspace(pin(3), dr(3), 4)'];
figure()
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials{keep_track},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{keep_track}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');

        elseif IC_answer_train{keep_track}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end

saveas( gcf , strcat(save_directory_PCA,'Single_Trials_',sess, '_', num2str(2) ) );

grad = [linspace(cy(1), dbl(1), 4)', linspace(cy(2), dbl(2), 4)',linspace(cy(3), dbl(3), 4)'];
figure()
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials{keep_track},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{keep_track}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');

        elseif IC_answer_train{keep_track}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end

saveas( gcf , strcat(save_directory_PCA,'Single_Trials_',sess, '_', num2str(3) ) );

grad = [linspace(cy(1), dbl(1), 4)', linspace(cy(2), dbl(2), 4)',linspace(cy(3), dbl(3), 4)'];
figure()
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials{keep_track},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{keep_track}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');

        elseif IC_answer_train{keep_track}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end
saveas( gcf , strcat(save_directory_PCA,'Single_Trials_',sess, '_', num2str(4) ) );

grad = [linspace(gld(1), dor(1), 4)', linspace(gld(2), dor(2), 4)',linspace(gld(3), dor(3), 4)'];
figure()
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials{keep_track},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{keep_track}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');

        elseif IC_answer_train{keep_track}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end

saveas( gcf , strcat(save_directory_PCA,'Single_Trials_',sess, '_', num2str(5) ) );

grad = [linspace(gld(1), dor(1), 4)', linspace(gld(2), dor(2), 4)',linspace(gld(3), dor(3), 4)'];
figure()
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials{keep_track},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{keep_track}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');

        elseif IC_answer_train{keep_track}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
            scatter(1, trl_act(1,1), 50 ,[0.8 0.8 0.8], 'filled');
            scatter(first_len(keep_track), trl_act(1,first_len(keep_track)), 50 ,[0.5 0.5 0.5], 'filled');
            scatter(first_len(keep_track) + second_len(keep_track),trl_act(1,first_len(keep_track) + second_len(keep_track)), 50 ,'k', 'filled');
        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end

saveas( gcf , strcat(save_directory_PCA,'Single_Trials_',sess, '_', num2str(6) ) );


%% Single Trials

grad = [linspace(pin(1), dr(1), 4)', linspace(pin(2), dr(2), 4)',linspace(pin(3), dr(3), 4)'];
figure()
keep_track = 0;
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials{conds},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{conds}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials{conds}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{conds},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on


        elseif IC_answer_train{conds}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials{conds}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{conds},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on

        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end

grad = [linspace(pin(1), dr(1), 4)', linspace(pin(2), dr(2), 4)',linspace(pin(3), dr(3), 4)'];
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials{keep_track},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{keep_track}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on


        elseif IC_answer_train{keep_track}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on

        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end



grad = [linspace(cy(1), dbl(1), 4)', linspace(cy(2), dbl(2), 4)',linspace(cy(3), dbl(3), 4)'];
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials{keep_track},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{keep_track}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on


        elseif IC_answer_train{keep_track}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
        
        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end


grad = [linspace(cy(1), dbl(1), 4)', linspace(cy(2), dbl(2), 4)',linspace(cy(3), dbl(3), 4)'];
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials{keep_track},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{keep_track}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
       

        elseif IC_answer_train{keep_track}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
 
        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end


grad = [linspace(gld(1), dor(1), 4)', linspace(gld(2), dor(2), 4)',linspace(gld(3), dor(3), 4)'];
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials{keep_track},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{keep_track}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on


        elseif IC_answer_train{keep_track}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
      
        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end

grad = [linspace(gld(1), dor(1), 4)', linspace(gld(2), dor(2), 4)',linspace(gld(3), dor(3), 4)'];
% plot trial average
for conds = 1:4
    keep_track = keep_track +1;
%     myVideo = VideoWriter(strcat('Single_Trial_Decoding_',sess, '_',num2str(conds) ) );
%     myVideo.FrameRate = 1;
%     open(myVideo)
    for trls = 1:size(model_pred_L_R_trials{keep_track},1)   
%         shadedErrorBar(1:size(model_pred_L_R_trials{conds},2),model_pred_L_R_trials{conds}(trls,:,:), {@(x) nanmean(x, 3),@(x) nanstd(x,3)}, 'lineprops', {'Color','k'},'patchSaturation',0.075) 
        
        if IC_answer_train{keep_track}(trls,1,1) == 0
            subplot(1,2,1)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on


        elseif IC_answer_train{keep_track}(trls,1,1) == 1
            subplot(1,2,2)
            trl_act = nanmean(model_pred_L_R_trials{keep_track}(trls,:,:),3);
            plot(1:size(model_pred_L_R_trials{keep_track},2), trl_act, 'Color', grad(conds,:))
            xlabel('time')
            ylabel('Left vs. Right ( -1 vs 1)')
            hold on
       
        end

%         pause(0.01);
%         frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
    end
%     close(myVideo)
end


saveas( gcf , strcat(save_directory_PCA,'Single_Trials_All_',sess ) );

%% 


% plot overall decoder performance

overall_decoder_accur = [];
overall_decoder_accur_shuffle = [];

for conds = 1:24
    size(model_pred_accuracy{conds}(:,end));
    overall_decoder_accur = vertcat(overall_decoder_accur, nanmean(model_pred_accuracy{conds}(:,end)) );
    overall_decoder_accur_shuffle = vertcat(overall_decoder_accur_shuffle, nanmean(model_pred_accuracy_shuffle{conds}(:,end)) );

end

figure()

bar(1,mean(overall_decoder_accur),'r')
hold on
errorbar(1,mean(overall_decoder_accur),std(overall_decoder_accur),'b')
hold on
bar(2,mean(overall_decoder_accur_shuffle),'k')
hold on
errorbar(2,mean(overall_decoder_accur_shuffle),std(overall_decoder_accur_shuffle),'b')
ylim([0 1])

saveas( gcf , strcat(save_directory_PCA,'Decoding_Performance_',sess) );

save(strcat(save_directory_PCA,'_decoding_data_L_R_', sess, '.mat'), 'model_pred_L_R','-v7.3');
save(strcat(save_directory_PCA,'_decoding_data_performance_', sess, '.mat'), 'model_pred_accuracy','-v7.3');

save(strcat(save_directory_PCA,'_model_pred_L_R_trials_', sess, '.mat'), 'model_pred_L_R_trials','-v7.3');
save(strcat(save_directory_PCA,'_model_pred_L_R_trials_raw_', sess, '.mat'), 'model_pred_L_R_trials_raw','-v7.3');
save(strcat(save_directory_PCA,'_model_pred_L_R_trials_sigmoid_', sess, '.mat'), 'model_pred_L_R_trials_sigmoid','-v7.3');

save(strcat(save_directory_PCA,'_model_pred_L_R_trials_shuffle_', sess, '.mat'), 'model_pred_L_R_trials_shuffle','-v7.3');
save(strcat(save_directory_PCA,'_model_pred_L_R_trials_shuffle_raw_', sess, '.mat'), 'model_pred_L_R_trials_shuffle_raw','-v7.3');
save(strcat(save_directory_PCA,'_model_pred_L_R_trials_shuffle_sigmoid_', sess, '.mat'), 'model_pred_L_R_trials_shuffle_sigmoid','-v7.3');

save(strcat(save_directory_PCA,'_nrn_test_', sess, '.mat'), 'nrn_train_full','-v7.3');
save(strcat(save_directory_PCA,'_IC_answer_train_', sess, '.mat'), 'IC_answer_train','-v7.3');

save(strcat(save_directory_PCA,'_decoding_data_L_R_shuffle_', sess, '.mat'), 'model_pred_L_R_shuffle','-v7.3');
save(strcat(save_directory_PCA,'_decoding_data_performance_shuffle_', sess, '.mat'), 'model_pred_accuracy_shuffle','-v7.3');

save(strcat(save_directory_PCA,'_Model_Weights_', sess, '.mat'), 'Model','-v7.3');

save(strcat(save_directory_PCA,'_choice_per_trial_', sess, '.mat'), 'choice_per_trial','-v7.3');

%% Test on random geometries

bootstrap_mean = [];
shuffled_bootstrap_mean = [];


for iter = 1:10
    
    test_size = 0.5;    
    cv = cvpartition(size(trial_count_labels_cat,1),'HoldOut',test_size);
    idx = cv.test;
    
    % ensure equal L/R
    minLabelCount = min([length(trial_count_labels_cat((trial_count_labels_cat == 0) & ~idx)), length(trial_count_labels_cat((trial_count_labels_cat == 1) & ~idx))]);
    
    indices_0 = find((trial_count_labels_cat == 0) & ~idx);
    indices_1 = find((trial_count_labels_cat == 1) & ~idx);
    
    [sampledData_0, idx_0 ] = datasample(trial_count_labels_cat((trial_count_labels_cat == 0) & ~idx), minLabelCount, 1, 'Replace', false);
    [sampledData_1, idx_1 ] = datasample(trial_count_labels_cat((trial_count_labels_cat == 1) & ~idx), minLabelCount, 1, 'Replace', false);  
    
    unsampledIndices_0 = setdiff(1:size(trial_count_labels_cat((trial_count_labels_cat == 0) & ~idx)), idx_0);
    unsampledIndices_1 = setdiff(1:size(trial_count_labels_cat((trial_count_labels_cat == 1) & ~idx)), idx_1);
    
    idx(indices_0(unsampledIndices_0)) = 1;
    idx(indices_1(unsampledIndices_1)) = 1;
    
    originalSizes = cellfun(@(x) x(1) , cellfun(@size, label_train(1:24), 'UniformOutput', false));
    % Reshape the concatenated array back into the original arrays
    cellArrayReshaped = cell(size(label_train(1:24)));
    startIdx = 1;
    for i = 1:numel(label_train(1:24))
        cellSize = [originalSizes(i),1];  
        numElements = prod(cellSize);
        cellArrayReshaped{i} = reshape(idx(startIdx:startIdx+numElements-1), cellSize);
        startIdx = startIdx + numElements;
    end
    
    nrn_train_iter  = cellfun(@(x, idx) x(~idx, :, :), nrn_train(1:24), cellArrayReshaped, 'UniformOutput', false);
    nrn_label_iter = cellfun(@(x, idx) x(~idx, :), label_train(1:24), cellArrayReshaped, 'UniformOutput', false);
    
    flatCells = cellfun(@(x) reshape(permute(x, [2, 1, 3]), [], size(x,3)), nrn_train_iter, 'UniformOutput', false);
    nrn_train_iter_cat = vertcat(flatCells{:});
    
    flatCells = cellfun(@(x) reshape(x', [], size(x,3)),  nrn_label_iter, 'UniformOutput', false);
    nrn_label_iter_cat = vertcat(flatCells{:});
    
    Model = fitclinear([nrn_train_iter_cat],[nrn_label_iter_cat],...
    'Learner','logistic','Solver','sgd','Regularization','ridge','OptimizeLearnRate',true, ...
    'Lambda',best_lamda, 'Prior', 'uniform');
    
    val_pred = predict(Model, nrn_99);
    
    bootstrap_mean = vertcat(bootstrap_mean, mean((val_pred >= 0.5) == label_99)*100);
    
    
    sub_label_train = nrn_label_iter_cat;
    
    Model = fitclinear(nrn_train_iter_cat,sub_label_train(randperm(size(sub_label_train, 1))),...
    'Learner','logistic','Solver','sgd','Regularization','ridge','OptimizeLearnRate',true,...
    'Lambda',best_lamda,'Prior', 'uniform');
    
    val_pred = predict(Model,nrn_99);
    
    shuffled_bootstrap_mean = vertcat(shuffled_bootstrap_mean, mean((val_pred >= 0.5) == label_99)*100);
     


end

    
    figure()
    bar(1,mean(bootstrap_mean),'r')
    hold on
    errorbar(1,mean(bootstrap_mean),std(bootstrap_mean),'b')
    hold on
    bar(2,mean(shuffled_bootstrap_mean),'k')
    hold on
    errorbar(2,mean(shuffled_bootstrap_mean),std(shuffled_bootstrap_mean),'b')
    ylim([0 100])

    saveas( gcf , strcat(save_directory_PCA,'Decoding_Rand_Geos_Validation_',sess) );
    
    save(strcat(save_directory_PCA,'_random_geos_', sess, '.mat'), 'bootstrap_mean','-v7.3');
    save(strcat(save_directory_PCA,'_random_geos_shuffle_', sess, '.mat'), 'shuffled_bootstrap_mean','-v7.3');
    
end
