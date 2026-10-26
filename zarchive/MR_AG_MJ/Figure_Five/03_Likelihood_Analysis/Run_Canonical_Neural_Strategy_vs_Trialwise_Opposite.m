function Run_Canonical_Neural_Strategy_vs_Trialwise_Opposite(n_samples,n_label_shuffles)
% Compare each canonical trial's neurally assigned H/S strategy with the
% opposite H/S strategy. This is separate from the established Figure Five
% likelihood pipeline and writes only to its own output directory.

if nargin < 1
    n_samples = 10000;
end
if nargin < 2
    n_label_shuffles = 0;
end
assert(isscalar(n_samples) && isfinite(n_samples) && n_samples >= 1 && ...
    n_samples == round(n_samples), 'n_samples must be a positive integer.');
assert(isscalar(n_label_shuffles) && isfinite(n_label_shuffles) && ...
    n_label_shuffles >= 0 && n_label_shuffles == round(n_label_shuffles), ...
    'n_label_shuffles must be a nonnegative integer.');

close all; clc;
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
physiology_root = fullfile(project_root, 'Data', 'Physiology');
helper_root = fullfile(script_dir, 'Model_Helpers');
output_root = fullfile(script_dir, ...
    'Canonical_Neural_vs_Opposite_Outputs');
if ~isfolder(output_root), mkdir(output_root); end
addpath(helper_root, '-begin');
addpath(project_root, '-end');

monkeys = {'Faure','Nielsen'};
sessions = load_top_two_sessions(project_root);
results = struct;
shuffle_data = struct;
rng(45, 'twister');

for monkey_idx = 1:numel(monkeys)
    monkey = monkeys{monkey_idx};
    scalar = load_timing_scalar(monkey);
    results.(monkey) = struct('neural_nll', [], 'opposite_nll', [], ...
        'path', [], 'session', [], 'random_neural_nll', [], ...
        'random_opposite_nll', [], 'random_session', []);
    shuffle_data.(monkey)=cell(numel(sessions.(monkey)),1);

    for session_idx = 1:numel(sessions.(monkey))
        session_name = sessions.(monkey){session_idx};
        all_trials = load_session_trials(physiology_root, project_root, ...
            monkey, session_name);
        trials = subset_trials(all_trials, ~all_trials.is_random);
        fprintf('%s %s: simulating %d canonical trials with %d samples.\n', ...
            monkey, session_name, numel(trials.trial_id), n_samples);
        [prob_h, prob_s] = simulate_hs_models(trials, scalar, n_samples, ...
            monkey, session_name);

        neural_probability = prob_s;
        neural_probability(trials.strategy == 1,:) = ...
            prob_h(trials.strategy == 1,:);
        opposite_probability = prob_h;
        opposite_probability(trials.strategy == 1,:) = ...
            prob_s(trials.strategy == 1,:);

        neural_nll = chosen_nll(neural_probability, trials.choice);
        opposite_nll = chosen_nll(opposite_probability, trials.choice);
        keep = valid_nll_pair(neural_nll, opposite_nll);
        fprintf(['%s %s: retained %d/%d paired trials ' ...
            '(%d zero-probability pairs excluded).\n'], ...
            monkey, session_name, nnz(keep), numel(keep), nnz(~keep));

        results.(monkey).neural_nll = [results.(monkey).neural_nll; ...
            neural_nll(keep)];
        results.(monkey).opposite_nll = [results.(monkey).opposite_nll; ...
            opposite_nll(keep)];
        results.(monkey).path = [results.(monkey).path; trials.path(keep)];
        results.(monkey).session = [results.(monkey).session; ...
            repmat(session_idx, nnz(keep), 1)];

        % Match the established Figure Five RNG convention: random-geometry
        % simulations must not advance the canonical-session simulation stream.
        rng_state_after_canonical=rng;
        random_trials = subset_trials(all_trials, all_trials.is_random);
        fprintf('%s %s: simulating %d random trials with %d samples.\n', ...
            monkey, session_name, numel(random_trials.trial_id), n_samples);
        [random_prob_h, random_prob_s] = simulate_hs_models( ...
            random_trials, scalar, n_samples, monkey, [session_name '_random']);
        random_neural_probability = random_prob_s;
        random_neural_probability(random_trials.strategy == 1,:) = ...
            random_prob_h(random_trials.strategy == 1,:);
        random_opposite_probability = random_prob_h;
        random_opposite_probability(random_trials.strategy == 1,:) = ...
            random_prob_s(random_trials.strategy == 1,:);
        random_neural_nll = chosen_nll(random_neural_probability, ...
            random_trials.choice);
        random_opposite_nll = chosen_nll(random_opposite_probability, ...
            random_trials.choice);
        random_keep = valid_nll_pair(random_neural_nll,random_opposite_nll);
        fprintf(['%s %s random: retained %d/%d paired trials ' ...
            '(%d zero-probability pairs excluded).\n'], ...
            monkey,session_name,nnz(random_keep),numel(random_keep),nnz(~random_keep));
        results.(monkey).random_neural_nll = [ ...
            results.(monkey).random_neural_nll;random_neural_nll(random_keep)];
        results.(monkey).random_opposite_nll = [ ...
            results.(monkey).random_opposite_nll;random_opposite_nll(random_keep)];
        results.(monkey).random_session = [results.(monkey).random_session; ...
            repmat(session_idx,nnz(random_keep),1)];

        if n_label_shuffles > 0
            canonical_h_nll=chosen_nll(prob_h,trials.choice);
            canonical_s_nll=chosen_nll(prob_s,trials.choice);
            random_h_nll=chosen_nll(random_prob_h,random_trials.choice);
            random_s_nll=chosen_nll(random_prob_s,random_trials.choice);
            shuffle_data.(monkey){session_idx}=struct( ...
                'strategy',[trials.strategy;random_trials.strategy], ...
                'h_nll',[canonical_h_nll;random_h_nll], ...
                's_nll',[canonical_s_nll;random_s_nll], ...
                'path',[trials.path;nan(numel(random_trials.path),1)], ...
                'is_random',[false(numel(trials.path),1); ...
                    true(numel(random_trials.path),1)]);
        end
        rng(rng_state_after_canonical);
    end
end

summary=plot_canonical_neural_vs_opposite(results, monkeys, output_root, n_samples);
if n_label_shuffles > 0
    run_label_shuffle_control(shuffle_data,summary,monkeys,script_dir, ...
        n_label_shuffles);
end
fprintf('Finished. Figure saved in %s\n', output_root);
end

function sessions = load_top_two_sessions(project_root)
ranking_file = fullfile(project_root, 'Figure_Five', '01_Session_Ranking', ...
    'Outputs', 'all_session_mahalanobis_results.csv');
assert(isfile(ranking_file), 'Run Stage 01 first: missing %s.', ranking_file);
ranking = readtable(ranking_file, 'TextType', 'string');
sessions = struct;
for monkey = ["Faure","Nielsen"]
    rows = ranking.monkey == monkey & startsWith(ranking.status,"passed") & ...
        isfinite(ranking.cv_mahalanobis);
    current = sortrows(ranking(rows,:), 'cv_mahalanobis', 'descend');
    assert(height(current) >= 2, 'Fewer than two ranked sessions for %s.', monkey);
    sessions.(char(monkey)) = cellstr(current.session(1:2));
    fprintf('%s sessions: %s, %s.\n', monkey, current.session(1), current.session(2));
end
end

function trials = load_session_trials(physiology_root, project_root, ...
        monkey, session_name)
label_file = fullfile(project_root, 'Figure_Five', ...
    '02_Strategy_Clustering', 'Outputs', monkey, ...
    ['Random_Geometry_Strategy_Labels_' session_name '.mat']);
assert(isfile(label_file), 'Missing Stage 02 label file: %s.', label_file);
labels = load(label_file, 'fixed_trial_ids', 'fixed_strategy_labels', ...
    'random_trial_ids_final','random_strategy_labels');
strategy_ids = double(labels.fixed_trial_ids(:));
strategy_values = double(labels.fixed_strategy_labels(:));
assert(numel(strategy_ids) == numel(strategy_values) && ...
    numel(unique(strategy_ids)) == numel(strategy_ids) && ...
    all(ismember(strategy_values,[0 1])), ...
    'Invalid canonical labels: monkey=%s session=%s.', monkey, session_name);
random_ids=double(labels.random_trial_ids_final(:));
random_values=double(labels.random_strategy_labels(:));
assert(numel(random_ids)==numel(random_values) && ...
    numel(unique(random_ids))==numel(random_ids) && ...
    all(ismember(random_values,[0 1])), ...
    'Invalid random labels: monkey=%s session=%s.',monkey,session_name);

list_file = fullfile(physiology_root, 'Single_Trial_List', monkey, ...
    ['single_trial_list_' session_name '.mat']);
assert(isfile(list_file), 'Missing single-trial list: %s.', list_file);
limits = load(list_file, 'min_value', 'max_value');

behavior_file = fullfile(physiology_root, 'Behavioral_Data', monkey, ...
    [session_name '_good_trials_concat.mat']);
assert(isfile(behavior_file), 'Missing behavior file: %s.', behavior_file);
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

trials = empty_trial_struct;
trial_ids = unique(data.trial_indices_all, 'stable');
for trial_idx = 1:numel(trial_ids)
    trial_id = trial_ids(trial_idx);
    rows = data.trial_indices_all == trial_id;
    fade = unique_scalar(data.trial_fade(rows), 'trial_fade', ...
        monkey, session_name, trial_id);
    is_random = logical(unique_scalar(data.rand_geo(rows), 'rand_geo', ...
        monkey, session_name, trial_id));
    if trial_id < limits.min_value || trial_id > limits.max_value || fade ~= 0
        continue;
    end

    maze = unique_scalar(data.geo_type(rows), 'geo_type', ...
        monkey, session_name, trial_id);
    path = unique_scalar(data.path_type(rows), 'path_type', ...
        monkey, session_name, trial_id);
    if is_random
        if maze ~= -99 || path ~= -99, continue; end
        [found,label_idx]=ismember(trial_id,random_ids);
        assert(found,'Missing random neural label: monkey=%s session=%s trial_id=%g.', ...
            monkey,session_name,trial_id);
        strategy=random_values(label_idx);
    else
        if maze < 1 || maze > 6 || path < 1 || path > 24, continue; end
        [found,label_idx]=ismember(trial_id,strategy_ids);
        assert(found,'Missing canonical neural label: monkey=%s session=%s trial_id=%g.', ...
            monkey,session_name,trial_id);
        strategy=strategy_values(label_idx);
    end

    arm_values = nan(1,6);
    for arm_idx = 1:6
        arm_values(arm_idx) = unique_scalar( ...
            data.(sprintf('h%d',arm_idx))(rows), sprintf('h%d',arm_idx), ...
            monkey, session_name, trial_id);
    end
    velocity = unique_scalar(data.vel(rows), 'vel', ...
        monkey, session_name, trial_id);
    assert(velocity > 0, 'Invalid velocity: monkey=%s session=%s trial_id=%g.', ...
        monkey, session_name, trial_id);
    d1 = unique_scalar(data.LR(rows), 'LR', monkey, session_name, trial_id);
    d2 = unique_scalar(data.LR2(rows), 'LR2', monkey, session_name, trial_id);
    answers = nan(1,4);
    for answer_idx = 1:4
        answers(answer_idx) = unique_scalar( ...
            data.(sprintf('trial_answer%d',answer_idx))(rows), ...
            sprintf('trial_answer%d',answer_idx), monkey, session_name, trial_id);
    end
    assert(sum(logical(answers)) == 1, ...
        'Invalid answer flags: monkey=%s session=%s trial_id=%g.', ...
        monkey, session_name, trial_id);

    trials.trial_id(end+1,1) = trial_id;
    trials.maze(end+1,1) = maze;
    trials.path(end+1,1) = path;
    trials.h(end+1,:) = arm_values ./ velocity;
    trials.d1(end+1,1) = d1;
    trials.d2(end+1,1) = d2;
    trials.strategy(end+1,1) = strategy;
    trials.choice(end+1,:) = answer_to_spatial_choice(d1,d2,logical(answers));
    trials.is_random(end+1,1)=is_random;
end

assert(~isempty(trials.trial_id), ...
    'No qualifying trials: monkey=%s session=%s.', monkey, session_name);
assert(all(isfinite(trials.h),'all'), ...
    'Nonfinite arm time: monkey=%s session=%s.', monkey, session_name);
assert(isequal(sort(trials.trial_id(~trials.is_random)),sort(strategy_ids)), ...
    'Loaded canonical IDs differ from Stage 02 labels: monkey=%s session=%s.', ...
    monkey, session_name);
assert(isequal(sort(trials.trial_id(trials.is_random)),sort(random_ids)), ...
    'Loaded random IDs differ from Stage 02 labels: monkey=%s session=%s.', ...
    monkey, session_name);
end

function trials = empty_trial_struct
trials = struct('trial_id',[],'maze',[],'path',[],'h',zeros(0,6), ...
    'd1',[],'d2',[],'strategy',[],'choice',zeros(0,4), ...
    'is_random',false(0,1));
end

function selected = subset_trials(trials,mask)
mask=logical(mask(:));
assert(numel(mask)==numel(trials.trial_id) && any(mask),'Invalid trial subset.');
selected=struct;
fields=fieldnames(trials);
for field_idx=1:numel(fields)
    field=fields{field_idx}; values=trials.(field);
    assert(size(values,1)==numel(mask),'Trial field row mismatch: %s.',field);
    selected.(field)=values(mask,:);
end
end

function value = unique_scalar(values, field, monkey, session_name, trial_id)
values = unique(values);
assert(numel(values) == 1 && isfinite(values), ...
    'Invalid %s: monkey=%s session=%s trial_id=%g.', ...
    field, monkey, session_name, trial_id);
value = double(values);
end

function choice = answer_to_spatial_choice(d1,d2,answers)
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
choice = zeros(1,4);
choice(order(find(answers,1))) = 1;
end

function [prob_h,prob_s] = simulate_hs_models(trials,scalar,n_samples, ...
        monkey,session_name)
n_trials = numel(trials.trial_id);
counts_h = zeros(n_trials,4);
counts_s = zeros(n_trials,4);
h1=trials.h(:,1)'; h2=trials.h(:,2)'; h3=trials.h(:,3)';
h4=trials.h(:,4)'; h5=trials.h(:,5)'; h6=trials.h(:,6)';
d1=trials.d1'; d2=trials.d2';
true_first=h1; true_first(d1==1)=h4(d1==1);
true_second=h2;
true_second(d1==-1 & d2==-1)=h3(d1==-1 & d2==-1);
true_second(d1==1 & d2==1)=h5(d1==1 & d2==1);
true_second(d1==1 & d2==-1)=h6(d1==1 & d2==-1);
[truth1,truth2,truth3,truth4]=deal(trials.choice(:,1)', ...
    trials.choice(:,2)',trials.choice(:,3)',trials.choice(:,4)');

for sample_idx=1:n_samples
    one=max(true_first+randn(1,n_trials).*scalar.*true_first,eps);
    two=max(true_second+randn(1,n_trials).*scalar.*true_second,eps);
    [a1,a2,a3,a4]=Hierarchical_Model_Physiology( ...
        h1,h2,h3,h4,h5,h6,one,two,n_trials,scalar, ...
        truth1,truth2,truth3,truth4);
    counts_h=counts_h+[a1',a2',a3',a4'];
    [a1,a2,a3,a4]=Sequential_Model_Physiology( ...
        h1,h2,h3,h4,h5,h6,one,two,n_trials,scalar, ...
        truth1,truth2,truth3,truth4);
    counts_s=counts_s+[a1',a2',a3',a4'];
    if mod(sample_idx,1000)==0 || sample_idx==n_samples
        fprintf('%s %s: %d/%d simulations complete.\n', ...
            monkey,session_name,sample_idx,n_samples);
    end
end
prob_h=counts_h./n_samples;
prob_s=counts_s./n_samples;
for probabilities={prob_h,prob_s}
    current=probabilities{1};
    assert(all(isfinite(current),'all') && all(current>=0,'all') && ...
        all(current<=1,'all') && all(abs(sum(current,2)-1)<1e-10), ...
        'Invalid H/S probabilities: monkey=%s session=%s.',monkey,session_name);
end
end

function nll = chosen_nll(probability,choice)
nll=-log(sum(probability.*choice,2)+eps);
end

function keep = valid_nll_pair(first_nll,second_nll)
assert(isequal(size(first_nll),size(second_nll)),'Paired NLL size mismatch.');
keep=isfinite(first_nll)&isfinite(second_nll)& ...
    first_nll < -log(eps) & second_nll < -log(eps);
end

function summary = plot_canonical_neural_vs_opposite(results,monkeys,output_root,n_samples)
marker_map=struct('Faure','o','Nielsen','s');
color_map=struct('Faure',[0.45 0.20 0.70],'Nielsen',[0.15 0.55 0.45]);
summary=struct; limits_data=[];
for monkey_idx=1:numel(monkeys)
    monkey=monkeys{monkey_idx};
    x=results.(monkey).opposite_nll;
    y=results.(monkey).neural_nll;
    assert(numel(x)==numel(y) && ~isempty(x), ...
        'Invalid canonical NLL vectors: monkey=%s.',monkey);
    [~,trial_p,~,trial_stats]=ttest(y,x,'Tail','left');
    path_x=nan(24,1); path_y=nan(24,1); path_n=zeros(24,1);
    for path_id=1:24
        rows=results.(monkey).path==path_id;
        assert(any(rows),'No retained canonical trials: monkey=%s path=%d.', ...
            monkey,path_id);
        path_x(path_id)=mean(x(rows)); path_y(path_id)=mean(y(rows));
        path_n(path_id)=nnz(rows);
    end
    [~,path_p,~,path_stats]=ttest(path_y,path_x,'Tail','left');
    random_x=results.(monkey).random_opposite_nll;
    random_y=results.(monkey).random_neural_nll;
    assert(numel(random_x)==numel(random_y) && ~isempty(random_x), ...
        'Invalid random-geometry NLL vectors: monkey=%s.',monkey);
    [~,random_p,~,random_stats]=ttest(random_y,random_x,'Tail','left');
    combined_x=[path_x;mean(random_x)];
    combined_y=[path_y;mean(random_y)];
    [~,combined_p,~,combined_stats]=ttest(combined_y,combined_x,'Tail','left');
    summary.(monkey)=struct('path_x',path_x,'path_y',path_y,'path_n',path_n, ...
        'trial_p',trial_p,'trial_t',trial_stats.tstat,'trial_df',trial_stats.df, ...
        'path_p',path_p,'path_t',path_stats.tstat,'path_df',path_stats.df, ...
        'mean_x',mean(path_x),'mean_y',mean(path_y), ...
        'random_x',mean(random_x),'random_y',mean(random_y), ...
        'random_p',random_p,'random_t',random_stats.tstat, ...
        'random_df',random_stats.df,'random_n',numel(random_y), ...
        'combined_p',combined_p,'combined_t',combined_stats.tstat, ...
        'combined_df',combined_stats.df,'combined_mean_x',mean(combined_x), ...
        'combined_mean_y',mean(combined_y));
    limits_data=[limits_data;path_x;path_y;mean(random_x);mean(random_y)]; %#ok<AGROW>
    fprintf(['%s canonical, neural < trial-wise opposite: trial-level paired ' ...
        't(%d)=%.6f, p=%.10g, n=%d; path-level paired t(%d)=%.6f, ' ...
        'p=%.10g, n=24; mean difference=%+.6f.\n'], ...
        monkey,trial_stats.df,trial_stats.tstat,trial_p,numel(y), ...
        path_stats.df,path_stats.tstat,path_p,mean(y-x));
    fprintf(['%s random pooled, neural < trial-wise opposite: paired ' ...
        't(%d)=%.6f, p=%.10g, n=%d trials; mean difference=%+.6f.\n'], ...
        monkey,random_stats.df,random_stats.tstat,random_p,numel(random_y), ...
        mean(random_y-random_x));
    fprintf(['%s overall 24 canonical paths plus pooled random condition, ' ...
        'neural < opposite: paired t(%d)=%.6f, p=%.10g, n=25 summaries; ' ...
        'mean difference=%+.6f.\n'],monkey,combined_stats.df, ...
        combined_stats.tstat,combined_p,mean(combined_y-combined_x));
end

lo=min(limits_data); hi=max(limits_data);
padding=.08*max(eps,hi-lo); limits=[lo-padding hi+padding];
fig=figure('Color','w','Position',[160 120 900 760]);
ax=axes(fig); hold(ax,'on');
plot(ax,limits,limits,'k--','LineWidth',1.5);
all_counts=[summary.Faure.path_n;summary.Nielsen.path_n];
count_min=min(all_counts); count_max=max(all_counts);
for monkey_idx=1:numel(monkeys)
    monkey=monkeys{monkey_idx}; item=summary.(monkey);
    color=color_map.(monkey);
    for path_id=1:24
        marker_size=30+100*(item.path_n(path_id)-count_min)/ ...
            max(1,count_max-count_min);
        scatter(ax,item.path_x(path_id),item.path_y(path_id),marker_size, ...
            marker_map.(monkey),'filled','MarkerFaceColor',color, ...
            'MarkerEdgeColor','k','LineWidth',.7);
    end
    scatter(ax,item.combined_mean_x,item.combined_mean_y,260,marker_map.(monkey), ...
        'filled','MarkerFaceColor',color,'MarkerEdgeColor','k','LineWidth',2.5);
    scatter(ax,item.random_x,item.random_y,220,'d','filled', ...
        'MarkerFaceColor',color,'MarkerEdgeColor','k','LineWidth',1.8);
end
xlabel(ax,'Opposite neural H/S mean trial-wise -log likelihood');
ylabel(ax,'Neural-label H/S mean trial-wise -log likelihood');
title(ax,sprintf(['Canonical paths and pooled random geometries: neural strategy versus trial-wise opposite ' ...
    '(%d simulations)'],n_samples));
xlim(ax,limits); ylim(ax,limits); axis(ax,'square'); grid(ax,'on'); box(ax,'on');
h_unity=plot(ax,nan,nan,'k--','LineWidth',1.5);
h_f=scatter(ax,nan,nan,70,'o','filled','MarkerFaceColor',color_map.Faure, ...
    'MarkerEdgeColor','k');
h_n=scatter(ax,nan,nan,70,'s','filled','MarkerFaceColor',color_map.Nielsen, ...
    'MarkerEdgeColor','k');
h_mean=scatter(ax,nan,nan,260,'o','filled','MarkerFaceColor',[.75 .75 .75], ...
    'MarkerEdgeColor','k','LineWidth',2.5);
h_random=scatter(ax,nan,nan,220,'d','filled','MarkerFaceColor',[.75 .75 .75], ...
    'MarkerEdgeColor','k','LineWidth',1.8);
legend(ax,[h_unity,h_f,h_n,h_mean,h_random], ...
    {'Unity','Faure paths','Nielsen paths','Monkey mean: 24 paths + random', ...
    'Random geometries pooled'}, ...
    'Location','southeast');
annotation=sprintf([ ...
    'One-sided tests (neural < opposite)\n' ...
    'Overall (24 paths + random): Faure p=%.3g; Nielsen p=%.3g\n' ...
    'Canonical paths: Faure p=%.3g; Nielsen p=%.3g\n' ...
    'Random trials: Faure p=%.3g; Nielsen p=%.3g'], ...
    summary.Faure.combined_p,summary.Nielsen.combined_p, ...
    summary.Faure.path_p,summary.Nielsen.path_p, ...
    summary.Faure.random_p,summary.Nielsen.random_p);
text(ax,limits(1)+.03*range(limits),limits(2)-.04*range(limits), ...
    annotation,'VerticalAlignment','top','FontSize',10, ...
    'BackgroundColor','w','Margin',5);
output_file=fullfile(output_root, ...
    'Canonical_Neural_Strategy_vs_Trialwise_Opposite.png');
exportgraphics(fig,output_file,'Resolution',300);
savefig(fig,strrep(output_file,'.png','.fig'));
close(fig);
fprintf('Saved %s\n',output_file);
end

function run_label_shuffle_control(shuffle_data,observed,monkeys,script_dir, ...
        n_label_shuffles)
output_root=fullfile(script_dir,'Neural_Label_Shuffle_Control_Outputs');
if ~isfolder(output_root), mkdir(output_root); end
rng(145,'twister');
null=struct;
for monkey_idx=1:numel(monkeys)
    monkey=monkeys{monkey_idx};
    null.(monkey).canonical=nan(n_label_shuffles,1);
    null.(monkey).random=nan(n_label_shuffles,1);
    null.(monkey).overall=nan(n_label_shuffles,1);
end

for shuffle_idx=1:n_label_shuffles
    for monkey_idx=1:numel(monkeys)
        monkey=monkeys{monkey_idx};
        path_x=cell(24,1); path_y=cell(24,1); random_x=[]; random_y=[];
        sessions=shuffle_data.(monkey);
        for session_idx=1:numel(sessions)
            item=sessions{session_idx};
            n_trials=numel(item.strategy);
            assert(n_trials>0 && numel(item.h_nll)==n_trials && ...
                numel(item.s_nll)==n_trials && numel(item.path)==n_trials && ...
                numel(item.is_random)==n_trials, ...
                'Invalid shuffle data: monkey=%s session=%d.',monkey,session_idx);
            shuffled_strategy=item.strategy(randperm(n_trials));
            keep=valid_nll_pair(item.h_nll,item.s_nll);
            h=item.h_nll(keep); s=item.s_nll(keep);
            labels=shuffled_strategy(keep);
            neural=s; neural(labels==1)=h(labels==1);
            opposite=h; opposite(labels==1)=s(labels==1);
            current_random=item.is_random(keep);
            current_path=item.path(keep);
            random_x=[random_x;opposite(current_random)]; %#ok<AGROW>
            random_y=[random_y;neural(current_random)]; %#ok<AGROW>
            for path_id=1:24
                rows=~current_random & current_path==path_id;
                path_x{path_id}=[path_x{path_id};opposite(rows)];
                path_y{path_id}=[path_y{path_id};neural(rows)];
            end
        end
        mean_x=nan(24,1); mean_y=nan(24,1);
        for path_id=1:24
            assert(~isempty(path_x{path_id}) && numel(path_x{path_id})==numel(path_y{path_id}), ...
                'Empty shuffled path: monkey=%s path=%d.',monkey,path_id);
            mean_x(path_id)=mean(path_x{path_id});
            mean_y(path_id)=mean(path_y{path_id});
        end
        assert(~isempty(random_x) && numel(random_x)==numel(random_y), ...
            'Invalid shuffled random trials: monkey=%s.',monkey);
        null.(monkey).canonical(shuffle_idx)=mean(mean_y-mean_x);
        null.(monkey).random(shuffle_idx)=mean(random_y-random_x);
        null.(monkey).overall(shuffle_idx)=mean([mean_y-mean_x; ...
            mean(random_y-random_x)]);
    end
    if mod(shuffle_idx,1000)==0 || shuffle_idx==n_label_shuffles
        fprintf('Neural-label shuffle control: %d/%d permutations complete.\n', ...
            shuffle_idx,n_label_shuffles);
    end
end

permutation_p=struct;
for monkey_idx=1:numel(monkeys)
    monkey=monkeys{monkey_idx};
    observed_canonical=mean(observed.(monkey).path_y-observed.(monkey).path_x);
    observed_random=observed.(monkey).random_y-observed.(monkey).random_x;
    observed_overall=observed.(monkey).combined_mean_y- ...
        observed.(monkey).combined_mean_x;
    permutation_p.(monkey).canonical=(1+nnz(null.(monkey).canonical<=observed_canonical)) ...
        /(n_label_shuffles+1);
    permutation_p.(monkey).random=(1+nnz(null.(monkey).random<=observed_random)) ...
        /(n_label_shuffles+1);
    permutation_p.(monkey).overall=(1+nnz(null.(monkey).overall<=observed_overall)) ...
        /(n_label_shuffles+1);
    fprintf(['%s within-session neural-label permutation test (lower tail): ' ...
        'overall observed 24-path-plus-random difference=%+.6f, p=%.10g; ' ...
        'canonical observed mean path difference=%+.6f, p=%.10g; ' ...
        'random observed pooled difference=%+.6f, p=%.10g.\n'], ...
        monkey,observed_overall,permutation_p.(monkey).overall, ...
        observed_canonical,permutation_p.(monkey).canonical, ...
        observed_random,permutation_p.(monkey).random);
end

plot_shuffle_nulls(null,observed,permutation_p,monkeys,output_root, ...
    n_label_shuffles);
end

function plot_shuffle_nulls(null,observed,p_value,monkeys,output_root,n_shuffles)
fig=figure('Color','w','Position',[100 100 1100 520]);
tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
for monkey_idx=1:numel(monkeys)
    monkey=monkeys{monkey_idx};
    observed_value=observed.(monkey).combined_mean_y- ...
        observed.(monkey).combined_mean_x;
    ax=nexttile; hold(ax,'on');
    histogram(ax,null.(monkey).overall,40, ...
        'FaceColor',[.65 .65 .65],'EdgeColor','none');
    xline(ax,observed_value,'r-','LineWidth',2.5);
    xline(ax,0,'k--','LineWidth',1.2);
    xlabel(ax,'Mean neural-label NLL minus opposite NLL');
    ylabel(ax,'Permutations');
    title(ax,sprintf(['%s — 24 path means + pooled random mean\n' ...
        'observed=%.3g, lower-tail p=%.3g'],monkey,observed_value, ...
        p_value.(monkey).overall));
    box(ax,'on'); grid(ax,'on');
end
sgtitle(fig,sprintf(['Within-session neural-label shuffle control ' ...
    '(%d permutations; combined canonical and random statistic)'],n_shuffles));
output_file=fullfile(output_root,'Neural_Label_Shuffle_Null_Distributions.png');
exportgraphics(fig,output_file,'Resolution',300);
savefig(fig,strrep(output_file,'.png','.fig')); close(fig);
fprintf('Saved %s\n',output_file);
end
