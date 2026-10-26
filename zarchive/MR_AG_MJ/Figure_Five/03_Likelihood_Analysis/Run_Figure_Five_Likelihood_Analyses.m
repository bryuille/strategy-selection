function Run_Figure_Five_Likelihood_Analyses(n_samples)
% Evaluate neurally assigned H/S and Figure 1 behavioral models on the
% canonical six-maze and held-out random-geometry physiology trials.
%
% Figure 1 parameters are never refit to physiology choices. For models
% with five behavioral fits, choice probabilities are averaged across the
% five fitted parameter sets before trial likelihoods are evaluated.
% Every qualifying random trial is retained, including low-confidence and
% OOD assignments. Random trials are pooled within monkey for plotting.

if nargin < 1
    n_samples = 10000;
end
assert(isscalar(n_samples) && n_samples >= 1 && n_samples == round(n_samples), ...
    'n_samples must be a positive integer.');
maze_alternative_only = false;

close all; clc;
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
physiology_root = fullfile(project_root, 'Data', 'Physiology');
benchmark_helpers = fullfile(script_dir, 'Model_Helpers');
parameter_root = fullfile(project_root, 'Data', 'Pre_Physiology', ...
    'Model_Results', 'Fitted_Parameters');
output_root = fullfile(script_dir, 'Outputs');
if ~isfolder(output_root), mkdir(output_root); end

addpath(benchmark_helpers, '-begin');
addpath(project_root, '-end');

model_fields = {'Total_Time', 'Optimal_Lapse', 'Revision_X_LR_Max'};
model_labels = {'Total Time', 'Optimal + Lapse', ...
    'Hierarchical + Revision'};
monkeys = {'Faure', 'Nielsen'};
sessions = load_top_two_sessions(project_root);

agg = initialize_aggregate(monkeys, model_fields);
rng(45, 'twister');

for monkey_idx = 1:numel(monkeys)
    monkey = monkeys{monkey_idx};
    scalar = load_timing_scalar(monkey);
    lapse_parameters = load_parameter_fits(parameter_root, monkey, ...
        'Optimal_Lapse', 1);
    revision_parameters = load_parameter_fits(parameter_root, monkey, ...
        'Revision_X_LR_Max', 3);
    print_fitted_parameters(monkey, lapse_parameters, revision_parameters);

    for session_idx = 1:numel(sessions.(monkey))
        session_name = sessions.(monkey){session_idx};
        fprintf('\n%s %s: loading canonical and held-out random trials.\n', ...
            monkey, session_name);
        trials = load_session_trials(physiology_root, project_root, ...
            monkey, session_name);

        canonical_trials = subset_trials(trials, ~trials.is_random);
        random_trials = subset_trials(trials, trials.is_random);


        fprintf('%s %s: simulating %d canonical trials with %d samples.\n', ...
            monkey, session_name, numel(canonical_trials.trial_id), n_samples);
        probabilities = simulate_models(canonical_trials, scalar, lapse_parameters, ...
            revision_parameters, n_samples, monkey, session_name);

        chosen = canonical_trials.choice;
        neural_nll = chosen_nll(probabilities.Neural_Strategy, chosen);
        maze_alternative_nll = compute_maze_alternative_nll(probabilities, ...
            canonical_trials, monkey);
        maze_expected_nll = compute_maze_expected_nll(probabilities, ...
            canonical_trials, monkey);
        assert(all(isfinite(neural_nll)), ...
            'Nonfinite neural-model NLL: monkey=%s session=%s.', ...
            monkey, session_name);
        assert(all(isfinite(maze_alternative_nll)), ...
            'Nonfinite maze-alternative NLL: monkey=%s session=%s.', ...
            monkey, session_name);
        assert(all(isfinite(maze_expected_nll)), ...
            'Nonfinite maze-expected NLL: monkey=%s session=%s.', ...
            monkey, session_name);

        expected_strategy = expected_strategy_by_maze(canonical_trials.maze, monkey);
        valid_violation = valid_nll_pair(neural_nll, maze_expected_nll);
        violation_rows = canonical_trials.strategy ~= expected_strategy & valid_violation;
        for mode_cell = {'corr','err'}
            mode = mode_cell{1};
            if strcmp(mode,'corr')
                mode_rows = canonical_trials.correct;
            else
                mode_rows = ~canonical_trials.correct;
            end
            for path_id = 1:24
                rows = canonical_trials.path == path_id & violation_rows & mode_rows;
                if ~any(rows), continue; end
                agg.(monkey).Violation.(mode).Neural{path_id} = [ ...
                    agg.(monkey).Violation.(mode).Neural{path_id}; neural_nll(rows)];
                agg.(monkey).Violation.(mode).Expected{path_id} = [ ...
                    agg.(monkey).Violation.(mode).Expected{path_id}; maze_expected_nll(rows)];
            end
        end

        if ~maze_alternative_only
            model_nll = struct;
            for model_idx = 1:numel(model_fields)
                field = model_fields{model_idx};
                model_nll.(field) = chosen_nll(probabilities.(field), chosen);
                assert(all(isfinite(model_nll.(field))), ...
                    'Nonfinite %s NLL: monkey=%s session=%s.', ...
                    field, monkey, session_name);
            end
        end

        for path_id = 1:24
            rows = canonical_trials.path == path_id;
            if ~any(rows), continue; end
            % Match the established Figure 5 summary filter: retain every
            % trial in the relevant maze/path comparison except one where
            % either fixed H/S model gave the observed choice probability
            % exactly zero (NLL = -log(eps)).
            fixed_rows = rows & maze_alternative_nll < -log(eps) & ...
                maze_expected_nll < -log(eps);
            agg.(monkey).Maze_Alternative{path_id} = [ ...
                agg.(monkey).Maze_Alternative{path_id}; maze_alternative_nll(fixed_rows)];
            agg.(monkey).Maze_Expected{path_id} = [ ...
                agg.(monkey).Maze_Expected{path_id}; maze_expected_nll(fixed_rows)];
            if ~maze_alternative_only
                for model_idx = 1:numel(model_fields)
                    field = model_fields{model_idx};
                    % Apply the zero-probability rule symmetrically to this
                    % specific neural-versus-Figure-1-model comparison.
                    pair_rows = rows & valid_nll_pair(neural_nll, model_nll.(field));
                    agg.(monkey).ModelPairs.(field).Neural{path_id} = [ ...
                        agg.(monkey).ModelPairs.(field).Neural{path_id}; neural_nll(pair_rows)];
                    agg.(monkey).ModelPairs.(field).Comparator{path_id} = [ ...
                        agg.(monkey).ModelPairs.(field).Comparator{path_id}; model_nll.(field)(pair_rows)];
                end
            end
        end

        % Preserve the canonical simulation stream exactly. Random trials
        % use an independent deterministic stream and cannot perturb the
        % next canonical session's random draws. The maze-alternative
        % figure itself includes the pooled random-geometry comparison.
        canonical_rng_state = rng;
        rng(10000 + 100*monkey_idx + session_idx, 'twister');
        fprintf('%s %s: simulating %d random trials with %d samples.\n', ...
            monkey, session_name, numel(random_trials.trial_id), n_samples);
        random_probabilities = simulate_models(random_trials, scalar, ...
            lapse_parameters, revision_parameters, n_samples, monkey, ...
            [session_name '_random']);
        rng(canonical_rng_state);
        random_neural_nll = chosen_nll( ...
            random_probabilities.Neural_Strategy, random_trials.choice);
        random_opposite_nll = chosen_nll( ...
            random_probabilities.Opposite_Strategy, random_trials.choice);
        random_hs_keep = valid_nll_pair(random_neural_nll, random_opposite_nll);
        agg.(monkey).Random.Neural_Strategy = [ ...
            agg.(monkey).Random.Neural_Strategy; random_neural_nll(random_hs_keep)];
        agg.(monkey).Random.Opposite_Strategy = [ ...
            agg.(monkey).Random.Opposite_Strategy; random_opposite_nll(random_hs_keep)];
        agg.(monkey).Random.session = [agg.(monkey).Random.session; ...
            session_idx.*ones(nnz(random_hs_keep),1)];
        agg.(monkey).Random.strategy = [agg.(monkey).Random.strategy; ...
            random_trials.strategy(random_hs_keep)];
        agg.(monkey).ZeroFilter.Random_HS.total = ...
            agg.(monkey).ZeroFilter.Random_HS.total + numel(random_hs_keep);
        agg.(monkey).ZeroFilter.Random_HS.excluded = ...
            agg.(monkey).ZeroFilter.Random_HS.excluded + nnz(~random_hs_keep);
        if ~maze_alternative_only
            for model_idx = 1:numel(model_fields)
                field = model_fields{model_idx};
                random_model_nll = chosen_nll(random_probabilities.(field), random_trials.choice);
                random_pair_keep = valid_nll_pair(random_neural_nll, random_model_nll);
                agg.(monkey).ModelPairs.(field).RandomNeural = [ ...
                    agg.(monkey).ModelPairs.(field).RandomNeural; random_neural_nll(random_pair_keep)];
                agg.(monkey).ModelPairs.(field).RandomComparator = [ ...
                    agg.(monkey).ModelPairs.(field).RandomComparator; random_model_nll(random_pair_keep)];
                agg.(monkey).ZeroFilter.(field).random_total = ...
                    agg.(monkey).ZeroFilter.(field).random_total + numel(random_pair_keep);
                agg.(monkey).ZeroFilter.(field).random_excluded = ...
                    agg.(monkey).ZeroFilter.(field).random_excluded + nnz(~random_pair_keep);
            end
        end
        if ~maze_alternative_only
            for model_idx = 1:numel(model_fields)
                field = model_fields{model_idx};
                pair_keep = valid_nll_pair(neural_nll, model_nll.(field));
                agg.(monkey).ZeroFilter.(field).canonical_total = ...
                    agg.(monkey).ZeroFilter.(field).canonical_total + numel(pair_keep);
                agg.(monkey).ZeroFilter.(field).canonical_excluded = ...
                    agg.(monkey).ZeroFilter.(field).canonical_excluded + nnz(~pair_keep);
            end
        end
    end
end

plot_fixed_strategy_vs_maze_alternative(agg, monkeys, output_root, n_samples);
plot_random_neural_strategy_vs_trialwise_opposite(agg, monkeys, output_root, n_samples);
plot_violation_correct_error(agg, monkeys, output_root, n_samples);
print_zero_filter_summary(agg, monkeys, model_fields, maze_alternative_only);

if ~maze_alternative_only
    for model_idx = 1:numel(model_fields)
        plot_model_comparison(agg, monkeys, model_fields{model_idx}, ...
            model_labels{model_idx}, output_root, n_samples);
    end
end

fprintf('\nFinished. Figures saved in %s\n', output_root);
end

function sessions=load_top_two_sessions(project_root)
ranking_file=fullfile(project_root,'Figure_Five','01_Session_Ranking', ...
    'Outputs','all_session_mahalanobis_results.csv');
assert(isfile(ranking_file),'Run Stage 01 first: missing %s.',ranking_file);
ranking=readtable(ranking_file,'TextType','string');
sessions=struct;
for monkey=["Faure","Nielsen"]
    rows=ranking.monkey==monkey & startsWith(ranking.status,"passed") & ...
        isfinite(ranking.cv_mahalanobis);
    current=sortrows(ranking(rows,:),'cv_mahalanobis','descend');
    assert(height(current)>=2,'Fewer than two ranked sessions for %s.',monkey);
    sessions.(char(monkey))=cellstr(current.session(1:2));
end
end

function print_fitted_parameters(monkey, lapse_parameters, revision_parameters)
fprintf('\n============================================================\n');
fprintf('%s: independently fitted Figure 1 parameters\n', monkey);
fprintf('Optimal + Lapse (fit, lapse rate):\n');
for fit_idx = 1:numel(lapse_parameters)
    fprintf('  fit %d: %.10g\n', fit_idx, lapse_parameters(fit_idx));
end
fprintf(['Hierarchical + Revision (fit, noise slope, additive noise, ' ...
    'revision threshold):\n']);
for fit_idx = 1:size(revision_parameters, 1)
    fprintf('  fit %d: %.10g, %.10g, %.10g\n', fit_idx, ...
        revision_parameters(fit_idx, 1), ...
        revision_parameters(fit_idx, 2), ...
        revision_parameters(fit_idx, 3));
end
fprintf('============================================================\n');
end

function agg = initialize_aggregate(monkeys, model_fields)
agg = struct;
for monkey_idx = 1:numel(monkeys)
    monkey = monkeys{monkey_idx};
    agg.(monkey).Maze_Alternative = cell(24, 1);
    agg.(monkey).Maze_Expected = cell(24, 1);
    agg.(monkey).Random = struct('Neural_Strategy', [], ...
        'Opposite_Strategy', [], ...
        'session', [], 'strategy', []);
    for mode_cell={'corr','err'}
        mode=mode_cell{1};
        agg.(monkey).Violation.(mode).Neural=cell(24,1);
        agg.(monkey).Violation.(mode).Expected=cell(24,1);
    end
    for model_idx = 1:numel(model_fields)
        field = model_fields{model_idx};
        agg.(monkey).ModelPairs.(field) = struct('Neural', {cell(24,1)}, ...
            'Comparator', {cell(24,1)}, 'RandomNeural', [], 'RandomComparator', []);
        agg.(monkey).ZeroFilter.(field) = struct('canonical_total', 0, ...
            'canonical_excluded', 0, 'random_total', 0, 'random_excluded', 0);
    end
    agg.(monkey).ZeroFilter.Random_HS = struct('total', 0, 'excluded', 0);
end
end

function parameters = load_parameter_fits(parameter_root, monkey, model, width)
parameters = nan(5, width);
for fit_idx = 1:5
    parameter_file = fullfile(parameter_root, monkey, model, ...
        sprintf('est_parameters_%d.mat', fit_idx));
    assert(isfile(parameter_file), 'Missing fitted parameters: %s', parameter_file);
    loaded = load(parameter_file, 'est_parameters');
    assert(isfield(loaded, 'est_parameters') && ...
        numel(loaded.est_parameters) == width, ...
        'Invalid fitted parameters in %s.', parameter_file);
    parameters(fit_idx, :) = reshape(double(loaded.est_parameters), 1, []);
end
assert(all(isfinite(parameters), 'all'), ...
    'Nonfinite fitted parameters for %s %s.', monkey, model);
end

function trials = load_session_trials(physiology_root, project_root, ...
        monkey, session_name)
% All four validated top-two label files live with this final analysis.
label_root = fullfile(project_root, 'Figure_Five', ...
    '02_Strategy_Clustering', 'Outputs');
random_label_file = fullfile(label_root, monkey, ...
    ['Random_Geometry_Strategy_Labels_' session_name '.mat']);
assert(isfile(random_label_file), 'Missing validated label file: %s', random_label_file);
random_labels = load(random_label_file, 'fixed_trial_ids', ...
    'fixed_strategy_labels', 'random_trial_ids_final', ...
    'random_strategy_labels');
strategy_ids = double(random_labels.fixed_trial_ids(:));
strategy_values = double(random_labels.fixed_strategy_labels(:));
assert(numel(strategy_ids) == numel(strategy_values) && ...
    numel(unique(strategy_ids)) == numel(strategy_ids) && ...
    all(ismember(strategy_values, [0 1])), ...
    'Invalid fixed strategy labels in %s.', random_label_file);
random_ids = double(random_labels.random_trial_ids_final(:));
random_values = double(random_labels.random_strategy_labels(:));
assert(numel(random_ids) == numel(random_values) && ...
    numel(unique(random_ids)) == numel(random_ids) && ...
    all(ismember(random_values,[0 1])), ...
    'Invalid random label file: %s.', random_label_file);

list_file = fullfile(physiology_root, 'Single_Trial_List', monkey, ...
    ['single_trial_list_' session_name '.mat']);
limits = load(list_file, 'min_value', 'max_value');

behavior_file = fullfile(physiology_root, 'Behavioral_Data', monkey, ...
    [session_name '_good_trials_concat.mat']);
fields = {'trial_fade','photodiode_qc_bad','h1','h2','h3','h4','h5','h6', ...
    'vel','LR','LR2','trial_answer1','trial_answer2','trial_answer3', ...
    'trial_answer4','trial_indices_all','geo_type','path_type','rand_geo'};
data = struct;
for field_idx = 1:numel(fields)
    field = fields{field_idx};
    data.(field) = h5read(behavior_file, ['/save_all_data/' field]);
    data.(field) = data.(field)(:);
end
good_rows = ~logical(data.photodiode_qc_bad);
for field_idx = 1:numel(fields)
    field = fields{field_idx};
    data.(field) = data.(field)(good_rows);
end

trial_ids = unique(data.trial_indices_all, 'stable');
trials = empty_trial_struct;
for trial_idx = 1:numel(trial_ids)
    trial_id = trial_ids(trial_idx);
    trial_rows = data.trial_indices_all == trial_id;
    geo = unique_scalar(data.geo_type(trial_rows), 'geo_type', monkey, ...
        session_name, trial_id);
    path = unique_scalar(data.path_type(trial_rows), 'path_type', monkey, ...
        session_name, trial_id);
    fade = unique_scalar(data.trial_fade(trial_rows), 'trial_fade', monkey, ...
        session_name, trial_id);
    is_random = logical(unique_scalar(data.rand_geo(trial_rows), 'rand_geo', ...
        monkey, session_name, trial_id));
    if trial_id < limits.min_value || trial_id > limits.max_value || fade ~= 0
        continue;
    end

    if is_random
        if geo ~= -99 || path ~= -99, continue; end
        [found, label_idx] = ismember(trial_id, random_ids);
        assert(found, ...
            'Missing random strategy label: monkey=%s session=%s trial_id=%g.', ...
            monkey, session_name, trial_id);
        strategy = random_values(label_idx);
    else
        if geo < 1 || geo > 6 || path < 1 || path > 24, continue; end
        [found, label_idx] = ismember(trial_id, strategy_ids);
        assert(found, ...
            'Missing canonical strategy label: monkey=%s session=%s trial_id=%g.', ...
            monkey, session_name, trial_id);
        strategy = strategy_values(label_idx);
    end
    arm_values = nan(1, 6);
    for arm_idx = 1:6
        arm_values(arm_idx) = unique_scalar(data.(sprintf('h%d', arm_idx))(trial_rows), ...
            sprintf('h%d', arm_idx), monkey, session_name, trial_id);
    end
    velocity = unique_scalar(data.vel(trial_rows), 'vel', monkey, session_name, trial_id);
    assert(isfinite(velocity) && velocity > 0, ...
        'Invalid velocity: monkey=%s session=%s trial_id=%g.', ...
        monkey, session_name, trial_id);
    d1 = unique_scalar(data.LR(trial_rows), 'LR', monkey, session_name, trial_id);
    d2 = unique_scalar(data.LR2(trial_rows), 'LR2', monkey, session_name, trial_id);
    answers = nan(1, 4);
    for answer_idx = 1:4
        answers(answer_idx) = unique_scalar( ...
            data.(sprintf('trial_answer%d', answer_idx))(trial_rows), ...
            sprintf('trial_answer%d', answer_idx), monkey, session_name, trial_id);
    end
    assert(sum(logical(answers)) == 1, ...
        'Invalid answer flags: monkey=%s session=%s trial_id=%g.', ...
        monkey, session_name, trial_id);

    choice = answer_to_spatial_choice(d1, d2, logical(answers));
    trials.trial_id(end+1, 1) = trial_id;
    trials.maze(end+1, 1) = geo;
    trials.path(end+1, 1) = path;
    trials.h(end+1, :) = arm_values ./ velocity;
    trials.d1(end+1, 1) = d1;
    trials.d2(end+1, 1) = d2;
    trials.strategy(end+1, 1) = strategy;
    trials.correct(end+1, 1) = logical(answers(1));
    trials.choice(end+1, :) = choice;
    trials.is_random(end+1, 1) = is_random;
end
assert(~isempty(trials.trial_id), ...
    'No qualifying trials: monkey=%s session=%s.', monkey, session_name);
assert(all(isfinite(trials.h), 'all'), ...
    'Nonfinite arm time: monkey=%s session=%s.', monkey, session_name);
loaded_random_ids = trials.trial_id(trials.is_random);
assert(isequal(sort(loaded_random_ids), sort(random_ids)), ...
    'Loaded random trial IDs differ from neural-label file: monkey=%s session=%s.', ...
    monkey, session_name);
fprintf('%s %s: %d canonical, %d random trials; all random trials included.\n', ...
    monkey, session_name, nnz(~trials.is_random), nnz(trials.is_random));
end

function trials = empty_trial_struct
trials = struct('trial_id', [], 'maze', [], 'path', [], 'h', zeros(0, 6), ...
    'd1', [], 'd2', [], 'strategy', [], 'correct', false(0,1), ...
    'choice', zeros(0, 4), ...
    'is_random', false(0,1));
end

function selected = subset_trials(trials, mask)
mask = logical(mask(:));
assert(numel(mask)==numel(trials.trial_id) && any(mask), ...
    'Invalid or empty trial subset.');
selected = struct;
fields = fieldnames(trials);
for field_idx = 1:numel(fields)
    field = fields{field_idx};
    values = trials.(field);
    assert(size(values,1)==numel(mask), ...
        'Trial field row mismatch for %s.', field);
    selected.(field) = values(mask,:);
end
end

function value = unique_scalar(values, field, monkey, session_name, trial_id)
values = unique(values);
assert(numel(values) == 1 && isfinite(values), ...
    'Invalid %s: monkey=%s session=%s trial_id=%g.', ...
    field, monkey, session_name, trial_id);
value = double(values);
end

function choice = answer_to_spatial_choice(d1, d2, answers)
% Preserve the exact answer-relative-to-spatial mapping used in Figure 5.
if d1 == -1 && d2 == 1
    order = [1 2 3 4];
elseif d1 == -1 && d2 == -1
    order = [2 1 4 3];
elseif d1 == 1 && d2 == 1
    order = [3 4 1 2];
elseif d1 == 1 && d2 == -1
    order = [4 3 2 1];
else
    error('Invalid LR/LR2 values: %g, %g.', d1, d2);
end
choice = zeros(1, 4);
choice(order(find(answers, 1))) = 1;
end

function probabilities = simulate_models(trials, scalar, lapse_parameters, ...
        revision_parameters, n_samples, monkey, session_name)
n_trials = numel(trials.trial_id);
counts_h = zeros(n_trials, 4);
counts_s = zeros(n_trials, 4);
counts_o = zeros(n_trials, 4);
counts_t = zeros(n_trials, 4);
counts_r = zeros(n_trials, 4, size(revision_parameters, 1));

h1 = trials.h(:, 1)'; h2 = trials.h(:, 2)'; h3 = trials.h(:, 3)';
h4 = trials.h(:, 4)'; h5 = trials.h(:, 5)'; h6 = trials.h(:, 6)';
d1 = trials.d1'; d2 = trials.d2';
true_first = h1;
true_first(d1 == 1) = h4(d1 == 1);
true_second = h2;
true_second(d1 == -1 & d2 == -1) = h3(d1 == -1 & d2 == -1);
true_second(d1 == 1 & d2 == 1) = h5(d1 == 1 & d2 == 1);
true_second(d1 == 1 & d2 == -1) = h6(d1 == 1 & d2 == -1);
true_total = true_first + true_second;
[truth1, truth2, truth3, truth4] = deal( ...
    trials.choice(:,1)', trials.choice(:,2)', ...
    trials.choice(:,3)', trials.choice(:,4)');

for sample_idx = 1:n_samples
    one = max(true_first + randn(1, n_trials) .* scalar .* true_first, eps);
    two = max(true_second + randn(1, n_trials) .* scalar .* true_second, eps);
    total_measure = max(true_total + randn(1, n_trials) .* scalar .* true_total, eps);

    [a1,a2,a3,a4] = Hierarchical_Model_Physiology(h1,h2,h3,h4,h5,h6, ...
        one,two,n_trials,scalar,truth1,truth2,truth3,truth4);
    counts_h = counts_h + [a1',a2',a3',a4'];
    [a1,a2,a3,a4] = Sequential_Model_Physiology(h1,h2,h3,h4,h5,h6, ...
        one,two,n_trials,scalar,truth1,truth2,truth3,truth4);
    counts_s = counts_s + [a1',a2',a3',a4'];
    [a1,a2,a3,a4] = Optimal_Lapse_Model_Physiology(h1,h2,h3,h4,h5,h6, ...
        one,two,n_trials,scalar,truth1,truth2,truth3,truth4);
    counts_o = counts_o + [a1',a2',a3',a4'];
    [a1,a2,a3,a4] = Total_Time_Model_Physiology(h1,h2,h3,h4,h5,h6, ...
        total_measure,n_trials,scalar,truth1,truth2,truth3,truth4);
    counts_t = counts_t + [a1',a2',a3',a4'];

    for fit_idx = 1:size(revision_parameters, 1)
        params = revision_parameters(fit_idx, :);
        noisy_sd = params(2) + params(1) .* scalar .* true_second;
        assert(all(isfinite(noisy_sd) & noisy_sd > 0), ...
            'Invalid revision noise: monkey=%s session=%s fit=%d.', ...
            monkey, session_name, fit_idx);
        two_noise = max(true_second + randn(1, n_trials) .* noisy_sd, eps);
        [a1,a2,a3,a4] = Revision_X_LR_Max_Model_Physiology( ...
            h1,h2,h3,h4,h5,h6,one,two,n_trials,params(3),scalar, ...
            truth1,truth2,truth3,truth4,params(1),params(2),two_noise);
        counts_r(:,:,fit_idx) = counts_r(:,:,fit_idx) + [a1',a2',a3',a4'];
    end

    if mod(sample_idx, 1000) == 0 || sample_idx == n_samples
        fprintf('%s %s: %d/%d simulations complete.\n', ...
            monkey, session_name, sample_idx, n_samples);
    end
end

prob_h = counts_h ./ n_samples;
prob_s = counts_s ./ n_samples;
prob_o = counts_o ./ n_samples;
probabilities.Hierarchical = prob_h;
probabilities.Sequential = prob_s;
probabilities.Total_Time = counts_t ./ n_samples;
probabilities.Optimal_Lapse = Apply_Optimal_Lapse_Parameters_Physiology( ...
    prob_o, lapse_parameters);
probabilities.Revision_X_LR_Max = mean(counts_r ./ n_samples, 3);
probabilities.Neural_Strategy = prob_s;
probabilities.Neural_Strategy(trials.strategy == 1, :) = ...
    prob_h(trials.strategy == 1, :);
probabilities.Opposite_Strategy = prob_h;
probabilities.Opposite_Strategy(trials.strategy == 1, :) = ...
    prob_s(trials.strategy == 1, :);

probability_fields = fieldnames(probabilities);
for field_idx = 1:numel(probability_fields)
    field = probability_fields{field_idx};
    prob = probabilities.(field);
    assert(all(isfinite(prob), 'all') && all(prob >= 0, 'all') && ...
        all(prob <= 1, 'all'), ...
        'Invalid probabilities for %s: monkey=%s session=%s.', ...
        field, monkey, session_name);
    assert(all(abs(sum(prob, 2) - 1) < 1e-10), ...
        'Probabilities do not sum to one for %s: monkey=%s session=%s.', ...
        field, monkey, session_name);
end
end

function nll = compute_maze_alternative_nll(probabilities, trials, monkey)
% Canonical comparison matches the original Figure 5 fixed-strategy logic:
% use Sequential as the alternative for expected-H mazes and Hierarchical
% as the alternative for expected-S mazes.
if strcmp(monkey, 'Faure')
    expected_hierarchical = 1:3;
elseif strcmp(monkey, 'Nielsen')
    expected_hierarchical = 1:4;
else
    error('Unknown monkey: %s.', monkey);
end
is_expected_h = ismember(trials.maze, expected_hierarchical);
assert(all(trials.maze >= 1 & trials.maze <= 6), ...
    'Maze-alternative comparison requires canonical maze IDs 1--6.');
alternative_probability = probabilities.Hierarchical;
alternative_probability(is_expected_h, :) = ...
    probabilities.Sequential(is_expected_h, :);
nll = chosen_nll(alternative_probability, trials.choice);
end

function nll = compute_maze_expected_nll(probabilities, trials, monkey)
if strcmp(monkey, 'Faure')
    expected_hierarchical = 1:3;
elseif strcmp(monkey, 'Nielsen')
    expected_hierarchical = 1:4;
else
    error('Unknown monkey: %s.', monkey);
end
is_expected_h = ismember(trials.maze, expected_hierarchical);
expected_probability = probabilities.Sequential;
expected_probability(is_expected_h, :) = ...
    probabilities.Hierarchical(is_expected_h, :);
nll = chosen_nll(expected_probability, trials.choice);
end

function expected = expected_strategy_by_maze(maze, monkey)
if strcmp(monkey, 'Faure')
    expected_hierarchical = 1:3;
elseif strcmp(monkey, 'Nielsen')
    expected_hierarchical = 1:4;
else
    error('Unknown monkey: %s.', monkey);
end
assert(all(maze >= 1 & maze <= 6), ...
    'Expected-strategy lookup requires canonical maze IDs 1--6.');
expected = double(ismember(maze, expected_hierarchical));
end

function nll = chosen_nll(probabilities, choice)
chosen_probability = sum(probabilities .* choice, 2);
nll = -log(chosen_probability + eps);
end

function keep = valid_nll_pair(first_nll, second_nll)
% A zero empirical choice probability is encoded as -log(eps) by
% chosen_nll. Remove it symmetrically from the paired comparison.
assert(isequal(size(first_nll), size(second_nll)), ...
    'Paired NLL size mismatch.');
keep = isfinite(first_nll) & isfinite(second_nll) & ...
    first_nll < -log(eps) & second_nll < -log(eps);
end

function print_zero_filter_summary(agg, monkeys, model_fields, maze_alternative_only)
fprintf('\nZero-probability NLL exclusion summary (retained / eligible):\n');
for monkey_idx = 1:numel(monkeys)
    monkey = monkeys{monkey_idx};
    hs = agg.(monkey).ZeroFilter.Random_HS;
    fprintf('  %s random neural vs opposite: %d / %d retained (%d excluded).\n', ...
        monkey, hs.total-hs.excluded, hs.total, hs.excluded);
    if maze_alternative_only, continue; end
    for model_idx = 1:numel(model_fields)
        field = model_fields{model_idx};
        stats = agg.(monkey).ZeroFilter.(field);
        fprintf(['  %s neural vs %s: canonical %d / %d retained (%d excluded); ' ...
            'random %d / %d retained (%d excluded).\n'], ...
            monkey, field, stats.canonical_total-stats.canonical_excluded, ...
            stats.canonical_total, stats.canonical_excluded, ...
            stats.random_total-stats.random_excluded, stats.random_total, ...
            stats.random_excluded);
    end
end
end



function plot_model_comparison(agg, monkeys, model_field, model_label, ...
        output_root, n_samples)
marker_map = struct('Faure', 'o', 'Nielsen', 's');
hier_color = [0.20 0.40 0.80];
seq_color = [0.90 0.50 0.10];
random_color = [0.55 0.25 0.70];
x_all = []; y_all = [];
summary = struct;

for monkey_idx = 1:numel(monkeys)
    monkey = monkeys{monkey_idx};
    summary.(monkey).x = nan(24,1);
    summary.(monkey).y = nan(24,1);
    summary.(monkey).n = zeros(24,1);
    for path_id = 1:24
        pair = agg.(monkey).ModelPairs.(model_field);
        x = pair.Comparator{path_id};
        y = pair.Neural{path_id};
        assert(numel(x) == numel(y), ...
            'Paired NLL length mismatch: model=%s monkey=%s path=%d.', ...
            model_field, monkey, path_id);
        if isempty(x), continue; end
        summary.(monkey).x(path_id) = mean(x);
        summary.(monkey).y(path_id) = mean(y);
        summary.(monkey).n(path_id) = numel(x);
    end
    valid = isfinite(summary.(monkey).x) & isfinite(summary.(monkey).y);
    x_all = [x_all; summary.(monkey).x(valid)]; %#ok<AGROW>
    y_all = [y_all; summary.(monkey).y(valid)]; %#ok<AGROW>
    [~, p, ~, stats] = ttest(summary.(monkey).y(valid), ...
        summary.(monkey).x(valid), 'Tail', 'left');
    fprintf('%s, Neural < %s, path-level paired t-test: t=%.4f, df=%d, p=%.6g, n=%d paths.\n', ...
        monkey, model_label, stats.tstat, stats.df, p, sum(valid));

    pair = agg.(monkey).ModelPairs.(model_field);
    random_x = pair.RandomComparator;
    random_y = pair.RandomNeural;
    assert(numel(random_x)==numel(random_y), ...
        'Random NLL length mismatch: model=%s monkey=%s.', model_field, monkey);
    summary.(monkey).random_x = mean(random_x);
    summary.(monkey).random_y = mean(random_y);
    delta_mean = mean(random_y-random_x);
    fprintf(['%s random pooled, Neural minus %s mean NLL = %.5f, ' ...
        'n=%d trials.\n'], monkey, model_label, delta_mean, numel(random_y));
    x_all = [x_all; summary.(monkey).random_x]; %#ok<AGROW>
    y_all = [y_all; summary.(monkey).random_y]; %#ok<AGROW>
end

limits_data = [x_all; y_all];
lo = min(limits_data); hi = max(limits_data);
padding = 0.06 * max(eps, hi - lo);
limits = [lo-padding, hi+padding];

fig = figure('Color', 'w', 'Position', [150 120 850 720]);
ax = axes(fig); hold(ax, 'on');
plot(ax, limits, limits, 'k--', 'LineWidth', 1.5);
all_counts = [];
for monkey_idx = 1:numel(monkeys)
    all_counts = [all_counts; summary.(monkeys{monkey_idx}).n]; %#ok<AGROW>
end
positive_counts = all_counts(all_counts > 0);
count_min = min(positive_counts); count_max = max(positive_counts);

for monkey_idx = 1:numel(monkeys)
    monkey = monkeys{monkey_idx};
    if strcmp(monkey, 'Faure'), hierarchical_mazes = 1:3;
    else, hierarchical_mazes = 1:4;
    end
    for path_id = 1:24
        x = summary.(monkey).x(path_id); y = summary.(monkey).y(path_id);
        if ~isfinite(x) || ~isfinite(y), continue; end
        maze = ceil(path_id / 4);
        if ismember(maze, hierarchical_mazes), color = hier_color;
        else, color = seq_color;
        end
        n = summary.(monkey).n(path_id);
        marker_size = 30 + 100 * (n-count_min) / max(1, count_max-count_min);
        scatter(ax, x, y, marker_size, marker_map.(monkey), 'filled', ...
            'MarkerFaceColor', color, 'MarkerEdgeColor', 'k', 'LineWidth', 0.7);
    end

    rx = summary.(monkey).random_x;
    ry = summary.(monkey).random_y;
    scatter(ax, rx, ry, 190, 'd', 'filled', ...
        'MarkerFaceColor', random_color, 'MarkerEdgeColor', 'k', 'LineWidth', 1.2);
end

xlabel(ax, sprintf('%s mean -log likelihood', model_label));
ylabel(ax, 'Neural-label H/S mean -log likelihood');
title(ax, sprintf('Neural strategy model versus %s (%d simulations)', ...
    model_label, n_samples));
xlim(ax, limits); ylim(ax, limits); axis(ax, 'square'); grid(ax, 'on'); box(ax, 'on');
h_unity = plot(ax, nan, nan, 'k--', 'LineWidth', 1.5);
h_h = scatter(ax, nan, nan, 70, 'o', 'filled', ...
    'MarkerFaceColor', hier_color, 'MarkerEdgeColor', 'k');
h_s = scatter(ax, nan, nan, 70, 'o', 'filled', ...
    'MarkerFaceColor', seq_color, 'MarkerEdgeColor', 'k');
h_f = scatter(ax, nan, nan, 70, 'o', 'MarkerEdgeColor', 'k');
h_n = scatter(ax, nan, nan, 70, 's', 'MarkerEdgeColor', 'k');
h_random = scatter(ax, nan, nan, 120, 'd', 'filled', ...
    'MarkerFaceColor', random_color, 'MarkerEdgeColor', 'k');
legend(ax, [h_unity,h_h,h_s,h_random,h_f,h_n], ...
    {'Unity','Expected hierarchical maze','Expected sequential maze', ...
    'Random geometry pooled','Faure','Nielsen'}, ...
    'Location', 'best');

output_file = fullfile(output_root, ...
    ['Neural_Strategy_vs_' model_field '_PerPath.png']);
exportgraphics(fig, output_file, 'Resolution', 300);
savefig(fig, strrep(output_file,'.png','.fig'));
fprintf('Saved %s\n', output_file);
close(fig);
end

function plot_fixed_strategy_vs_maze_alternative(agg, monkeys, output_root, n_samples)
% Exact canonical six-maze axis from the established Figure 5 plot:
% expected condition-defined H/S versus the opposite expected strategy.
marker_map = struct('Faure', 'o', 'Nielsen', 's');
hier_color = [0.20 0.40 0.80];
seq_color = [0.90 0.50 0.10];
summary = struct;
limits_data = [];

for monkey_idx = 1:numel(monkeys)
    monkey = monkeys{monkey_idx};
    summary.(monkey).x = nan(24,1);
    summary.(monkey).y = nan(24,1);
    summary.(monkey).n = zeros(24,1);
    for path_id = 1:24
        x = agg.(monkey).Maze_Alternative{path_id};
        y = agg.(monkey).Maze_Expected{path_id};
        assert(numel(x) == numel(y), ...
            'Canonical fixed-strategy NLL length mismatch: monkey=%s path=%d.', ...
            monkey, path_id);
        if isempty(x), continue; end
        summary.(monkey).x(path_id) = mean(x);
        summary.(monkey).y(path_id) = mean(y);
        summary.(monkey).n(path_id) = numel(x);
    end
    valid = isfinite(summary.(monkey).x) & isfinite(summary.(monkey).y);
    path_maze = ceil((1:24)' ./ 4);
    if strcmp(monkey,'Faure')
        expected_hierarchical = 1:3;
    else
        expected_hierarchical = 1:4;
    end
    h_paths = ismember(path_maze, expected_hierarchical);
    fprintf('%s fixed-comparison retained trials: expected-H=%d, expected-S=%d.\n', ...
        monkey, sum(summary.(monkey).n(h_paths)), sum(summary.(monkey).n(~h_paths)));
    [~,p,~,stats] = ttest(summary.(monkey).y(valid), ...
        summary.(monkey).x(valid), 'Tail', 'left');
    fprintf(['%s, Expected fixed H/S < maze-defined alternative, path-level paired ' ...
        't-test: t=%.4f, df=%d, p=%.6g, n=%d paths.\n'], ...
        monkey,stats.tstat,stats.df,p,nnz(valid));
    limits_data = [limits_data; summary.(monkey).x(valid); ...
        summary.(monkey).y(valid)]; %#ok<AGROW>
end

lo = min(limits_data); hi = max(limits_data);
padding = 0.06*max(eps,hi-lo);
limits = [lo-padding hi+padding];
fig = figure('Color','w','Position',[150 120 900 760]);
ax = axes(fig); hold(ax,'on');
plot(ax,limits,limits,'k--','LineWidth',1.5);
all_counts = vertcat(summary.(monkeys{1}).n,summary.(monkeys{2}).n);
positive_counts = all_counts(all_counts>0);
count_min = min(positive_counts); count_max = max(positive_counts);

for monkey_idx = 1:numel(monkeys)
    monkey = monkeys{monkey_idx};
    if strcmp(monkey,'Faure'), expected_hierarchical = 1:3;
    else, expected_hierarchical = 1:4;
    end
    for path_id = 1:24
        x = summary.(monkey).x(path_id);
        y = summary.(monkey).y(path_id);
        if ~isfinite(x) || ~isfinite(y), continue; end
        maze = ceil(path_id/4);
        if ismember(maze,expected_hierarchical), color = hier_color;
        else, color = seq_color;
        end
        n = summary.(monkey).n(path_id);
        size_value = 30 + 100*(n-count_min)/max(1,count_max-count_min);
        scatter(ax,x,y,size_value,marker_map.(monkey),'filled', ...
            'MarkerFaceColor',color,'MarkerEdgeColor','k','LineWidth',0.7);
    end
end

xlabel(ax,'Opposite expected H/S mean -log likelihood');
ylabel(ax,'Expected maze H/S mean -log likelihood');
title(ax,sprintf('Fixed strategy versus maze alternative (%d simulations)',n_samples));
xlim(ax,limits); ylim(ax,limits); axis(ax,'square'); grid(ax,'on'); box(ax,'on');
h_unity = plot(ax,nan,nan,'k--','LineWidth',1.5);
h_h = scatter(ax,nan,nan,70,'o','filled','MarkerFaceColor',hier_color,'MarkerEdgeColor','k');
h_s = scatter(ax,nan,nan,70,'o','filled','MarkerFaceColor',seq_color,'MarkerEdgeColor','k');
h_f = scatter(ax,nan,nan,70,'o','MarkerEdgeColor','k');
h_n = scatter(ax,nan,nan,70,'s','MarkerEdgeColor','k');
legend(ax,[h_unity,h_h,h_s,h_f,h_n], ...
    {'Unity','Expected hierarchical maze','Expected sequential maze', ...
    'Faure','Nielsen'}, ...
    'Location','best');
output_file = fullfile(output_root,'Fixed_Strategy_vs_Maze_Alternative_PerPath.png');
exportgraphics(fig,output_file,'Resolution',300);
savefig(fig,strrep(output_file,'.png','.fig'));
fprintf('Saved %s\n',output_file);
close(fig);
end

function plot_random_neural_strategy_vs_trialwise_opposite(agg, monkeys, output_root, n_samples)
marker_map = struct('Faure','o','Nielsen','s');
color_map = struct('Faure',[0.45 0.20 0.70],'Nielsen',[0.15 0.55 0.45]);
summary = struct; limits_data = [];
for monkey_idx = 1:numel(monkeys)
    monkey = monkeys{monkey_idx};
    x = agg.(monkey).Random.Opposite_Strategy;
    y = agg.(monkey).Random.Neural_Strategy;
    assert(numel(x)==numel(y) && ~isempty(x), ...
        'Invalid random NLL vectors: monkey=%s.',monkey);
    summary.(monkey).x = mean(x);
    summary.(monkey).y = mean(y);
    summary.(monkey).sem_x = std(x, 0) ./ sqrt(numel(x));
    summary.(monkey).sem_y = std(y, 0) ./ sqrt(numel(y));
    summary.(monkey).n = numel(y);
    [~, summary.(monkey).p_value, ~, stats] = ttest(y, x, 'Tail', 'left');
    summary.(monkey).t_stat = stats.tstat;
    summary.(monkey).df = stats.df;
    fprintf(['%s random pooled, neural minus trial-wise opposite mean NLL = %.5f, ' ...
        'paired one-sided t-test: t(%d)=%.4f, p=%.6g, n=%d trials.\n'], ...
        monkey, mean(y-x), stats.df, stats.tstat, summary.(monkey).p_value, numel(y));
    limits_data = [limits_data; x; y; ...
        summary.(monkey).x-summary.(monkey).sem_x; ...
        summary.(monkey).x+summary.(monkey).sem_x; ...
        summary.(monkey).y-summary.(monkey).sem_y; ...
        summary.(monkey).y+summary.(monkey).sem_y]; %#ok<AGROW>
end
lo = min(limits_data); hi = max(limits_data);
padding = 0.10*max(eps,hi-lo); limits = [lo-padding hi+padding];
fig = figure('Color','w','Position',[150 120 760 700]);
ax = axes(fig); hold(ax,'on'); plot(ax,limits,limits,'k--','LineWidth',1.5);
for monkey_idx = 1:numel(monkeys)
    monkey = monkeys{monkey_idx};
    x = agg.(monkey).Random.Opposite_Strategy;
    y = agg.(monkey).Random.Neural_Strategy;
    color = color_map.(monkey);
    scatter(ax, x, y, 38, marker_map.(monkey), 'filled', ...
        'MarkerFaceColor', color, 'MarkerEdgeColor', color, ...
        'MarkerFaceAlpha', 0.28, 'MarkerEdgeAlpha', 0.28);

    % Horizontal and vertical bars are SEM across retained random trials.
    plot(ax, [summary.(monkey).x-summary.(monkey).sem_x, ...
        summary.(monkey).x+summary.(monkey).sem_x], ...
        [summary.(monkey).y, summary.(monkey).y], '-', 'Color', color, 'LineWidth', 2.0);
    plot(ax, [summary.(monkey).x, summary.(monkey).x], ...
        [summary.(monkey).y-summary.(monkey).sem_y, ...
        summary.(monkey).y+summary.(monkey).sem_y], '-', 'Color', color, 'LineWidth', 2.0);
    scatter(ax,summary.(monkey).x,summary.(monkey).y,220,marker_map.(monkey), ...
        'filled','MarkerFaceColor',color,'MarkerEdgeColor','k','LineWidth',1.2);
end
xlabel(ax,'Trial-wise opposite neural H/S -log likelihood');
ylabel(ax,'Neural-label H/S -log likelihood');
title(ax,sprintf('Random geometries: neural strategy versus trial-wise opposite (%d simulations)',n_samples));
xlim(ax,limits); ylim(ax,limits); axis(ax,'square'); grid(ax,'on'); box(ax,'on');
h_unity = plot(ax,nan,nan,'k--','LineWidth',1.5);
h_f = scatter(ax,nan,nan,100,'o','MarkerEdgeColor','k');
h_n = scatter(ax,nan,nan,100,'s','MarkerEdgeColor','k');
legend(ax,[h_unity,h_f,h_n],{'Unity','Faure','Nielsen'},'Location','best');
annotation_text = sprintf(['One-sided paired t-tests (neural < opposite)\n' ...
    'Faure: t(%d) = %.2f, p = %.3g, n = %d\n' ...
    'Nielsen: t(%d) = %.2f, p = %.3g, n = %d'], ...
    summary.Faure.df, summary.Faure.t_stat, summary.Faure.p_value, summary.Faure.n, ...
    summary.Nielsen.df, summary.Nielsen.t_stat, summary.Nielsen.p_value, summary.Nielsen.n);
text(ax, limits(1) + 0.03*range(limits), limits(2) - 0.04*range(limits), ...
    annotation_text, 'VerticalAlignment','top', 'FontSize',10, ...
    'BackgroundColor','w', 'Margin',5);
output_file = fullfile(output_root,'Random_Neural_Strategy_vs_Trialwise_Opposite.png');
exportgraphics(fig,output_file,'Resolution',300);
savefig(fig,strrep(output_file,'.png','.fig'));
fprintf('Saved %s\n',output_file);
close(fig);
end

function plot_violation_correct_error(agg, monkeys, output_root, n_samples)
% Established violation display: per-path means, correct/error separated,
% monkey marker shapes, and one global mean for each monkey/outcome group.
modes={'corr','err'};
marker_map=struct('Faure','o','Nielsen','s');
color_map=struct('corr',[0.2 0.6 0.2],'err',[0.85 0.2 0.2]);
summary=struct;
pooled_expected=struct('corr',[],'err',[]);
pooled_neural=struct('corr',[],'err',[]);
for monkey_idx=1:numel(monkeys)
    monkey=monkeys{monkey_idx};
    for mode_idx=1:numel(modes)
        mode=modes{mode_idx}; x_mean=nan(24,1); y_mean=nan(24,1);
        n_path=zeros(24,1); x_all=[]; y_all=[];
        for path_id=1:24
            x=agg.(monkey).Violation.(mode).Expected{path_id};
            y=agg.(monkey).Violation.(mode).Neural{path_id};
            assert(numel(x)==numel(y), ...
                'Violation pair mismatch: monkey=%s mode=%s path=%d.', ...
                monkey,mode,path_id);
            if isempty(x), continue; end
            x_mean(path_id)=mean(x); y_mean(path_id)=mean(y);
            n_path(path_id)=numel(x); x_all=[x_all;x]; y_all=[y_all;y]; %#ok<AGROW>
        end
        assert(~isempty(x_all),'No violation trials: monkey=%s mode=%s.',monkey,mode);
        summary.(monkey).(mode)=struct('x',x_mean,'y',y_mean,'n',n_path, ...
            'global_x',mean(x_all),'global_y',mean(y_all));
        pooled_expected.(mode)=[pooled_expected.(mode);x_all];
        pooled_neural.(mode)=[pooled_neural.(mode);y_all];
        fprintf('%s %s violation paths plotted: %d.\n', ...
            monkey,mode,nnz(n_path>0));
    end
end
fprintf('\n=== POOLED ACROSS BOTH MONKEYS: TRIAL-LEVEL T-TESTS ===\n');
L_corr=min(numel(pooled_neural.corr),numel(pooled_expected.corr));
x_corr=pooled_neural.corr(1:L_corr);
y_corr=pooled_expected.corr(1:L_corr);
[~,p_corr_all,~,st_corr_all]=ttest(x_corr,y_corr,'Tail','left');
fprintf('CORRECT trials : t=%.6f, df=%d, p=%.6f (N=%d trials)\n', ...
    st_corr_all.tstat,st_corr_all.df,p_corr_all,L_corr);

L_err=min(numel(pooled_neural.err),numel(pooled_expected.err));
x_err=pooled_neural.err(1:L_err);
y_err=pooled_expected.err(1:L_err);
[~,p_err_all,~,st_err_all]=ttest(x_err,y_err,'Tail','left');
fprintf('ERROR trials   : t=%.6f, df=%d, p=%.10f (N=%d trials)\n', ...
    st_err_all.tstat,st_err_all.df,p_err_all,L_err);

pooled_neural_all=[pooled_neural.corr;pooled_neural.err];
pooled_expected_all=[pooled_expected.corr;pooled_expected.err];
L_all=min(numel(pooled_neural_all),numel(pooled_expected_all));
x_all=pooled_neural_all(1:L_all);
y_all=pooled_expected_all(1:L_all);
if L_all>1
    [~,p_all_trials,~,st_all_trials]=ttest(x_all,y_all,'Tail','left');
else
    p_all_trials=NaN;
    st_all_trials.tstat=NaN;
    st_all_trials.df=NaN;
end
fprintf('ALL trials (correct + error): t=%.6f, df=%d, p=%.6f (N=%d trials)\n\n', ...
    st_all_trials.tstat,st_all_trials.df,p_all_trials,L_all);
all_counts=[]; all_values=[];
for monkey_idx=1:numel(monkeys)
    monkey=monkeys{monkey_idx};
    for mode_idx=1:numel(modes)
        item=summary.(monkey).(modes{mode_idx}); valid=isfinite(item.x)&isfinite(item.y);
        all_counts=[all_counts;item.n(valid)]; %#ok<AGROW>
        all_values=[all_values;item.x(valid);item.y(valid);item.global_x;item.global_y]; %#ok<AGROW>
    end
end
all_counts=all_counts(all_counts>0); count_min=min(all_counts); count_max=max(all_counts);
lim_hi=prctile(all_values(isfinite(all_values)),99.5); limits=[0 lim_hi+0.05*(lim_hi+eps)];
fig=figure('Color','w','Position',[220 200 700 600]); ax=axes(fig); hold(ax,'on');
handles=gobjects(0); labels={};
for monkey_idx=1:numel(monkeys)
    monkey=monkeys{monkey_idx};
    for mode_idx=1:numel(modes)
        mode=modes{mode_idx}; item=summary.(monkey).(mode);
        valid=isfinite(item.x)&isfinite(item.y);
        sizes=20+120*(item.n(valid)-count_min)/max(1,count_max-count_min);
        handles(end+1)=scatter(ax,item.x(valid),item.y(valid),sizes, ...
            marker_map.(monkey),'filled','MarkerFaceColor',color_map.(mode), ...
            'MarkerEdgeColor','k'); %#ok<AGROW>
        labels{end+1}=[monkey ' ' mode]; %#ok<AGROW>
        handles(end+1)=scatter(ax,item.global_x,item.global_y,240, ...
            marker_map.(monkey),'filled','MarkerFaceColor',color_map.(mode), ...
            'MarkerEdgeColor','k','LineWidth',1.8); %#ok<AGROW>
        labels{end+1}=[monkey ' ' mode ' global mean']; %#ok<AGROW>
    end
end
plot(ax,limits,limits,'k--','LineWidth',1.5);
xlabel(ax,'Expected maze H/S mean trial-wise -log likelihood');
ylabel(ax,'Neural-label H/S mean trial-wise -log likelihood');
title(ax,sprintf(['Violation trials: neural versus expected strategy\n' ...
    '(correct/error separated; %d simulations)'],n_samples));
xlim(ax,limits); ylim(ax,limits); axis(ax,'square'); grid(ax,'on'); box(ax,'on');
legend(ax,handles,labels,'Location','best');
output_file=fullfile(output_root, ...
    'Violation_Neural_Strategy_vs_Condition_Expected_PerPath.png');
exportgraphics(fig,output_file,'Resolution',300);
savefig(fig,strrep(output_file,'.png','.fig')); close(fig);
fprintf('Saved %s\n',output_file);
end
