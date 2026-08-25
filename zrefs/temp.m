clear; close all; clc;

%% ============================================================
%  Batch eye movie QC, script version, no functions
%
%  Loops through all Eye_Data_<SESSION>.mat files.
%  Makes ONE random valid-trial QC movie per session.
%  Saves all movies into ONE shared folder.
%  Saves a CSV list of saved/skipped sessions.
% ============================================================

project_root = '/Volumes/Portable/MR_AG_MJ';

behavior_root = fullfile(project_root, 'Behavioral_Data');
eye_root      = fullfile(project_root, 'Eye_Data');

MONKEYS = {'Faure', 'Nielsen'};

% Optional filter.
% Leave empty to run all sessions.
SESSION_FILTER = 'june_22';

out_dir = fullfile(project_root, 'Eye_QC_Movies_AllSessions');
 
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

fprintf('Output dir: %s\n', out_dir);

%% Movie settings

N_RANDOM_TRIALS_PER_SESSION = 1;
RNG_SEED = 43;

FRAME_STEP_MS = 10;
VIDEO_FRAME_RATE = 100;

FLASH_DURATION_MS = 100;

PRE_FLASH1_MS = 1000;
POST_FLASH3_MS = 1000;

SHOW_FIGURE = true;

EYE_X_FIELD = 'eye_x';
EYE_Y_FIELD = 'eye_y';
PUPIL_FIELD = 'pupil_size';

GAZE_MARKER_SIZE = 28;
GAZE_TRAIL_MS = 150;
DRAW_GAZE_TRAIL = true;

MAX_EYE_SAMPLE_ERROR_MS = 20;

FLIP_EYE_Y = false;

EYE_X_OFFSET = 0;
EYE_Y_OFFSET = 0;

SKIP_GAZE_OUTSIDE_AXIS = false;

AX_LIMS = [-15 15 -15 15];

REMOVE_BAD_EYE_SAMPLES = true;
BAD_PUPIL_VALUE = -32768;

FIXATION_RADIUS_DEG = 1;
DRAW_FIXATION_CIRCLE = true;
DRAW_FIXATION_ONLY_BEFORE_FIX_OFF = false;

%% Output list

qc_movie_list = table( ...
    strings(0,1), ...
    strings(0,1), ...
    strings(0,1), ...
    strings(0,1), ...
    strings(0,1), ...
    nan(0,1), ...
    nan(0,1), ...
    strings(0,1), ...
    'VariableNames', { ...
        'monkey', ...
        'session', ...
        'eye_file', ...
        'behavior_file', ...
        'movie_file', ...
        'selected_trial_id', ...
        'n_valid_candidate_trials', ...
        'status'});

%% ============================================================
%  Main loop
% ============================================================

for m = 1:numel(MONKEYS)

    MONKEY = MONKEYS{m};

    behavior_directory = fullfile(behavior_root, MONKEY);
    eye_directory      = fullfile(eye_root, MONKEY);

    fprintf('\n============================================================\n');
    fprintf('Monkey: %s\n', MONKEY);
    fprintf('Eye dir: %s\n', eye_directory);
    fprintf('Behavior dir: %s\n', behavior_directory);
    fprintf('============================================================\n');

    if ~exist(eye_directory, 'dir')
        warning('Missing eye directory: %s', eye_directory);
        continue;
    end

    if ~exist(behavior_directory, 'dir')
        warning('Missing behavior directory: %s', behavior_directory);
        continue;
    end

    eye_files = dir(fullfile(eye_directory, 'Eye_Data_*.mat'));
    eye_files = eye_files(~startsWith({eye_files.name}, '.'));

    if ~isempty(SESSION_FILTER)
        eye_files = eye_files(contains({eye_files.name}, SESSION_FILTER));
    end

    [~, ord] = sort({eye_files.name});
    eye_files = eye_files(ord);

    fprintf('Eye sessions found: %d\n', numel(eye_files));

    for e = 1:numel(eye_files)

        clear D Eye_Data Eye_Load All_Data;
        clear eye_x eye_y eye_t_ms eye_y_t_ms pupil_size pupil_t_ms;
        clear v fig;

        eye_file = fullfile(eye_directory, eye_files(e).name);

        session_name = erase(eye_files(e).name, 'Eye_Data_');
        session_name = erase(session_name, '.mat');

        behavior_file = fullfile(behavior_directory, [session_name '_good_trials_concat.mat']);

        out_file = fullfile(out_dir, sprintf('%s_%s_eye_QC.mp4', MONKEY, session_name));

        fprintf('\n------------------------------------------------------------\n');
        fprintf('Session: %s %s\n', MONKEY, session_name);
        fprintf('Eye file:      %s\n', eye_file);
        fprintf('Behavior file: %s\n', behavior_file);
        fprintf('Output movie:  %s\n', out_file);

        status = "unknown";
        selected_trial_id_for_table = NaN;
        n_valid_trials_for_table = NaN;

        try

            if ~exist(behavior_file, 'file')

                warning('Missing behavior file. Skipping.');

                status = "skipped_missing_behavior_file";

                qc_movie_list(end+1,:) = { ...
                    string(MONKEY), ...
                    string(session_name), ...
                    string(eye_file), ...
                    string(behavior_file), ...
                    string(out_file), ...
                    selected_trial_id_for_table, ...
                    n_valid_trials_for_table, ...
                    string(status)};

                continue;

            end

            %% Load behavior

            All_Data = load(behavior_file);

            if ~isfield(All_Data, 'save_all_data')

                warning('Missing save_all_data. Skipping.');

                status = "skipped_missing_save_all_data";

                qc_movie_list(end+1,:) = { ...
                    string(MONKEY), ...
                    string(session_name), ...
                    string(eye_file), ...
                    string(behavior_file), ...
                    string(out_file), ...
                    selected_trial_id_for_table, ...
                    n_valid_trials_for_table, ...
                    string(status)};

                continue;

            end

            D = All_Data.save_all_data;

            %% Load eye

            Eye_Load = load(eye_file);

            if isfield(Eye_Load, 'Eye_Data')
                Eye_Data = Eye_Load.Eye_Data;
            else
                tmp_names = fieldnames(Eye_Load);
                Eye_Data = Eye_Load.(tmp_names{1});
            end

            %% Check behavior fields

            required_behavior_fields = { ...
                'trial_indices_all', ...
                'geo_present', ...
                'flash_one', ...
                'flash_two', ...
                'flash_three', ...
                'trial_end', ...
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6', ...
                'path_type'};

            missing_behavior_fields = {};

            for f = 1:numel(required_behavior_fields)
                if ~isfield(D, required_behavior_fields{f})
                    missing_behavior_fields{end+1} = required_behavior_fields{f}; %#ok<AGROW>
                end
            end

            if ~isempty(missing_behavior_fields)

                warning('Missing behavior fields: %s. Skipping.', strjoin(missing_behavior_fields, ', '));

                status = "skipped_missing_behavior_fields";

                qc_movie_list(end+1,:) = { ...
                    string(MONKEY), ...
                    string(session_name), ...
                    string(eye_file), ...
                    string(behavior_file), ...
                    string(out_file), ...
                    selected_trial_id_for_table, ...
                    n_valid_trials_for_table, ...
                    string(status)};

                continue;

            end

            %% Check eye fields

            required_eye_fields = {EYE_X_FIELD, EYE_Y_FIELD};

            missing_eye_fields = {};

            for f = 1:numel(required_eye_fields)
                if ~isfield(Eye_Data, required_eye_fields{f})
                    missing_eye_fields{end+1} = required_eye_fields{f}; %#ok<AGROW>
                end
            end

            if ~isempty(missing_eye_fields)

                warning('Missing eye fields: %s. Skipping.', strjoin(missing_eye_fields, ', '));

                status = "skipped_missing_eye_fields";

                qc_movie_list(end+1,:) = { ...
                    string(MONKEY), ...
                    string(session_name), ...
                    string(eye_file), ...
                    string(behavior_file), ...
                    string(out_file), ...
                    selected_trial_id_for_table, ...
                    n_valid_trials_for_table, ...
                    string(status)};

                continue;

            end

            has_pupil = isfield(Eye_Data, PUPIL_FIELD);

            %% Find valid candidate trials

            trial_ids_all = unique(D.trial_indices_all(:), 'stable');
            trial_ids_all = trial_ids_all(~isnan(trial_ids_all));
            trial_ids_all = trial_ids_all(trial_ids_all >= 1 & trial_ids_all <= numel(Eye_Data.(EYE_X_FIELD)));

            valid_trial_ids = [];

            for ii = 1:numel(trial_ids_all)

                tid = trial_ids_all(ii);

                row = find(D.trial_indices_all(:) == tid, 1, 'first');

                if isempty(row)
                    continue;
                end

                if tid > numel(Eye_Data.(EYE_X_FIELD)) || tid > numel(Eye_Data.(EYE_Y_FIELD))
                    continue;
                end

                if isempty(Eye_Data.(EYE_X_FIELD){tid}) || isempty(Eye_Data.(EYE_Y_FIELD){tid})
                    continue;
                end

                geo_s = D.geo_present(row);

                flash1_ms = (D.flash_one(row)   - geo_s) * 1000;
                flash2_ms = (D.flash_two(row)   - geo_s) * 1000;
                flash3_ms = (D.flash_three(row) - geo_s) * 1000;

                if any(isnan([flash1_ms flash2_ms flash3_ms])) || ...
                   flash1_ms <= 0 || flash2_ms <= flash1_ms || flash3_ms <= flash2_ms
                    continue;
                end

                E = double(Eye_Data.(EYE_X_FIELD){tid});

                if size(E, 2) == 2
                    eye_t_ms_tmp = E(:,2) * 1000;
                elseif size(E, 1) == 2
                    eye_t_ms_tmp = E(2,:)' * 1000;
                else
                    warning('Unexpected eye_x shape for trial %d. Skipping trial.', tid);
                    continue;
                end

                eye_t_ms_tmp = eye_t_ms_tmp(~isnan(eye_t_ms_tmp));

                if isempty(eye_t_ms_tmp)
                    continue;
                end

                movie_start_ms_candidate = max(min(eye_t_ms_tmp), flash1_ms - PRE_FLASH1_MS);
                movie_end_ms_candidate   = min(max(eye_t_ms_tmp), flash3_ms + POST_FLASH3_MS);

                if movie_end_ms_candidate <= movie_start_ms_candidate
                    continue;
                end

                if max(eye_t_ms_tmp) < flash3_ms
                    continue;
                end

                valid_trial_ids = vertcat(valid_trial_ids, tid); %#ok<AGROW>

            end

            n_valid_trials_for_table = numel(valid_trial_ids);

            fprintf('Valid candidate trials: %d\n', n_valid_trials_for_table);

            if isempty(valid_trial_ids)

                warning('No valid trials found. Skipping.');

                status = "skipped_no_valid_trials";

                qc_movie_list(end+1,:) = { ...
                    string(MONKEY), ...
                    string(session_name), ...
                    string(eye_file), ...
                    string(behavior_file), ...
                    string(out_file), ...
                    selected_trial_id_for_table, ...
                    n_valid_trials_for_table, ...
                    string(status)};

                continue;

            end

            %% Pick one random valid trial

            rng(RNG_SEED + m * 100000 + e, 'twister');

            if numel(valid_trial_ids) <= N_RANDOM_TRIALS_PER_SESSION
                selected_trial_ids = valid_trial_ids;
            else
                selected_trial_ids = valid_trial_ids(randperm(numel(valid_trial_ids), N_RANDOM_TRIALS_PER_SESSION));
            end

            tid = selected_trial_ids(1);
            selected_trial_id_for_table = tid;

            fprintf('Selected QC trial ID: %d\n', tid);

            row = find(D.trial_indices_all(:) == tid, 1, 'first');

            %% Load eye_x for selected trial

            E = double(Eye_Data.(EYE_X_FIELD){tid});

            if size(E, 2) == 2
                eye_x = E(:,1);
                eye_t_ms = E(:,2) * 1000;
            elseif size(E, 1) == 2
                eye_x = E(1,:)';
                eye_t_ms = E(2,:)' * 1000;
            else
                error('Unexpected eye_x data shape: %d x %d', size(E,1), size(E,2));
            end

            keep_x = ~isnan(eye_x) & ~isnan(eye_t_ms);
            eye_x = eye_x(keep_x);
            eye_t_ms = eye_t_ms(keep_x);

            %% Load eye_y for selected trial

            E = double(Eye_Data.(EYE_Y_FIELD){tid});

            if size(E, 2) == 2
                eye_y = E(:,1);
                eye_y_t_ms = E(:,2) * 1000;
            elseif size(E, 1) == 2
                eye_y = E(1,:)';
                eye_y_t_ms = E(2,:)' * 1000;
            else
                error('Unexpected eye_y data shape: %d x %d', size(E,1), size(E,2));
            end

            keep_y = ~isnan(eye_y) & ~isnan(eye_y_t_ms);
            eye_y = eye_y(keep_y);
            eye_y_t_ms = eye_y_t_ms(keep_y);

            if length(eye_y_t_ms) ~= length(eye_t_ms) || any(abs(eye_y_t_ms - eye_t_ms) > 1e-6)
                eye_y = interp1(eye_y_t_ms, eye_y, eye_t_ms, 'linear', NaN);
            end

            if FLIP_EYE_Y
                eye_y = -eye_y;
            end

            eye_x = eye_x + EYE_X_OFFSET;
            eye_y = eye_y + EYE_Y_OFFSET;

            %% Remove bad eye samples

            if REMOVE_BAD_EYE_SAMPLES

                bad_eye = false(size(eye_x));

                bad_eye = bad_eye | eye_x < AX_LIMS(1) | eye_x > AX_LIMS(2);
                bad_eye = bad_eye | eye_y < AX_LIMS(3) | eye_y > AX_LIMS(4);

                if has_pupil && tid <= numel(Eye_Data.(PUPIL_FIELD)) && ~isempty(Eye_Data.(PUPIL_FIELD){tid})

                    E = double(Eye_Data.(PUPIL_FIELD){tid});

                    if size(E, 2) == 2
                        pupil_size = E(:,1);
                        pupil_t_ms = E(:,2) * 1000;
                    elseif size(E, 1) == 2
                        pupil_size = E(1,:)';
                        pupil_t_ms = E(2,:)' * 1000;
                    else
                        pupil_size = [];
                        pupil_t_ms = [];
                    end

                    if ~isempty(pupil_size)

                        keep_p = ~isnan(pupil_size) & ~isnan(pupil_t_ms);
                        pupil_size = pupil_size(keep_p);
                        pupil_t_ms = pupil_t_ms(keep_p);

                        if numel(pupil_size) >= 2 && numel(pupil_t_ms) >= 2

                            bad_pupil_samples = pupil_size == BAD_PUPIL_VALUE;

                            bad_pupil_on_eye_time = interp1( ...
                                pupil_t_ms, ...
                                double(bad_pupil_samples), ...
                                eye_t_ms, ...
                                'nearest', ...
                                0);

                            bad_eye = bad_eye | bad_pupil_on_eye_time > 0;

                        end

                    end

                end

                eye_x(bad_eye) = NaN;
                eye_y(bad_eye) = NaN;

            end

            %% Event timing

            geo_s = D.geo_present(row);

            flash1_ms = (D.flash_one(row)   - geo_s) * 1000;
            flash2_ms = (D.flash_two(row)   - geo_s) * 1000;
            flash3_ms = (D.flash_three(row) - geo_s) * 1000;

            if isfield(D, 'fix_start')
                fix_start_ms = (D.fix_start(row) - geo_s) * 1000;
            else
                fix_start_ms = NaN;
            end

            if isfield(D, 'fixation_off')
                fixation_off_ms = (D.fixation_off(row) - geo_s) * 1000;
            else
                fixation_off_ms = NaN;
            end

            if isfield(D, 'answer_time')
                answer_ms = (D.answer_time(row) - geo_s) * 1000;
            else
                answer_ms = NaN;
            end

            if isfield(D, 'trial_end')
                trial_end_ms = (D.trial_end(row) - geo_s) * 1000;
            else
                trial_end_ms = NaN;
            end

            %% Print timing sanity check

            [~, idx_f1_eye] = min(abs(eye_t_ms - flash1_ms));
            [~, idx_f2_eye] = min(abs(eye_t_ms - flash2_ms));
            [~, idx_f3_eye] = min(abs(eye_t_ms - flash3_ms));

            fprintf('Photodiode-aligned eye sample check:\n');
            fprintf('  F1 target %.3f ms | closest eye %.3f ms | error %.3f ms\n', ...
                flash1_ms, eye_t_ms(idx_f1_eye), eye_t_ms(idx_f1_eye) - flash1_ms);
            fprintf('  F2 target %.3f ms | closest eye %.3f ms | error %.3f ms\n', ...
                flash2_ms, eye_t_ms(idx_f2_eye), eye_t_ms(idx_f2_eye) - flash2_ms);
            fprintf('  F3 target %.3f ms | closest eye %.3f ms | error %.3f ms\n', ...
                flash3_ms, eye_t_ms(idx_f3_eye), eye_t_ms(idx_f3_eye) - flash3_ms);

            if ~isnan(fixation_off_ms)
                [~, idx_fixoff_eye] = min(abs(eye_t_ms - fixation_off_ms));
                fprintf('  Fix off target %.3f ms | closest eye %.3f ms | error %.3f ms\n', ...
                    fixation_off_ms, eye_t_ms(idx_fixoff_eye), eye_t_ms(idx_fixoff_eye) - fixation_off_ms);
            end

            %% Maze geometry

            h1 = D.h1(row);
            h2 = D.h2(row);
            h3 = D.h3(row);
            h4 = D.h4(row);
            h5 = D.h5(row);
            h6 = D.h6(row);

            path_type = D.path_type(row);

            path_type_all = D.path_type(:);
            path_type_all = path_type_all(~isnan(path_type_all));

            if ~isempty(path_type_all) && min(path_type_all) == 0 && max(path_type_all) <= 23
                path_type_disp = path_type + 1;
            else
                path_type_disp = path_type;
            end

            %% Movie window

            movie_start_ms = max(min(eye_t_ms), flash1_ms - PRE_FLASH1_MS);
            movie_end_ms   = min(max(eye_t_ms), flash3_ms + POST_FLASH3_MS);

            if movie_end_ms <= movie_start_ms

                warning('Invalid movie window. Skipping.');

                status = "skipped_invalid_movie_window";

                qc_movie_list(end+1,:) = { ...
                    string(MONKEY), ...
                    string(session_name), ...
                    string(eye_file), ...
                    string(behavior_file), ...
                    string(out_file), ...
                    selected_trial_id_for_table, ...
                    n_valid_trials_for_table, ...
                    string(status)};

                continue;

            end

            frame_times_ms = movie_start_ms:FRAME_STEP_MS:movie_end_ms;

            %% Create video

            v = VideoWriter(out_file, 'MPEG-4');
            v.FrameRate = VIDEO_FRAME_RATE;
            open(v);

            if SHOW_FIGURE
                fig = figure('Color', 'white', 'Position', [100 100 750 750]);
            else
                fig = figure('Color', 'white', 'Position', [100 100 750 750], 'Visible', 'off');
            end

            %% Frame loop

            for ff = 1:numel(frame_times_ms)

                t_ms = frame_times_ms(ff);

                clf(fig);
                hold on;

                %% Draw H-maze

                midpoint = (h1 + h4) / 2;
                offset = -midpoint;

                left_x = offset;
                junction_x = h1 + offset;
                right_x = h1 + h4 + offset;

                plot([left_x junction_x], [0 0], 'k', 'LineWidth', 2);
                plot([junction_x right_x], [0 0], 'k', 'LineWidth', 2);

                plot([left_x left_x], [-h3 h2], 'k', 'LineWidth', 2);
                plot([right_x right_x], [-h6 h5], 'k', 'LineWidth', 2);

                plot([junction_x junction_x], [0 7], 'k', 'LineWidth', 2);

                plot(left_x, h2, 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'w');
                plot(left_x, -h3, 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'w');
                plot(right_x, h5, 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'w');
                plot(right_x, -h6, 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'w');

                plot(junction_x, 0, 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'k');

                axis equal;
                axis(AX_LIMS);
                set(gca, 'YDir', 'normal');
                set(gca, 'Color', 'white');
                box on;

                xlabel('x position');
                ylabel('y position');

                title(sprintf('%s %s | trial ID %d | path %d | t = %.0f ms', ...
                    MONKEY, session_name, tid, path_type_disp, t_ms), ...
                    'Interpreter', 'none');

                fixation_x = junction_x;
                fixation_y = 7;

                %% Fixation circle

                if DRAW_FIXATION_CIRCLE

                    if DRAW_FIXATION_ONLY_BEFORE_FIX_OFF
                        draw_fix_circle_now = isnan(fixation_off_ms) || t_ms <= fixation_off_ms;
                    else
                        draw_fix_circle_now = true;
                    end

                    if draw_fix_circle_now

                        theta = linspace(0, 2*pi, 200);
                        circ_x = fixation_x + FIXATION_RADIUS_DEG * cos(theta);
                        circ_y = fixation_y + FIXATION_RADIUS_DEG * sin(theta);

                        plot(circ_x, circ_y, '-', ...
                            'Color', [0 0.4 1], ...
                            'LineWidth', 2.0);

                        plot(fixation_x, fixation_y, '+', ...
                            'Color', [0 0.4 1], ...
                            'MarkerSize', 8, ...
                            'LineWidth', 1.5);

                    end

                end

                %% Moving ball / ball at junction

                if t_ms <= flash1_ms
                    ball_y = 7 - 7 * (t_ms / max(flash1_ms, eps));
                    ball_y = max(ball_y, 0);
                    plot(junction_x, ball_y, 'o', ...
                        'MarkerSize', 13, ...
                        'MarkerFaceColor', 'k', ...
                        'MarkerEdgeColor', 'k');
                else
                    plot(junction_x, 0, 'o', ...
                        'MarkerSize', 8, ...
                        'MarkerFaceColor', [0.3 0.3 0.3], ...
                        'MarkerEdgeColor', 'k');
                end

                %% Flash marker

                is_flash1 = t_ms >= flash1_ms && t_ms < flash1_ms + FLASH_DURATION_MS;
                is_flash2 = t_ms >= flash2_ms && t_ms < flash2_ms + FLASH_DURATION_MS;
                is_flash3 = t_ms >= flash3_ms && t_ms < flash3_ms + FLASH_DURATION_MS;

                if is_flash1 || is_flash2 || is_flash3

                    plot(junction_x, 0, 'o', ...
                        'MarkerSize', 42, ...
                        'MarkerFaceColor', [1 1 0.2], ...
                        'MarkerEdgeColor', 'k', ...
                        'LineWidth', 2);

                    if is_flash1
                        flash_label = 'FLASH 1';
                    elseif is_flash2
                        flash_label = 'FLASH 2';
                    else
                        flash_label = 'FLASH 3';
                    end

                    text(junction_x, 1.3, flash_label, ...
                        'HorizontalAlignment', 'center', ...
                        'FontSize', 16, ...
                        'FontWeight', 'bold', ...
                        'Color', 'k');

                end

                %% Fixation off text

                if ~isnan(fixation_off_ms) && abs(t_ms - fixation_off_ms) <= FRAME_STEP_MS
                    text(0, 13.5, 'FIXATION OFF', ...
                        'HorizontalAlignment', 'center', ...
                        'FontSize', 16, ...
                        'FontWeight', 'bold', ...
                        'Color', [0.6 0 0.8]);
                end

                %% Gaze trail

                if DRAW_GAZE_TRAIL

                    trail_idx = eye_t_ms >= (t_ms - GAZE_TRAIL_MS) & eye_t_ms <= t_ms;

                    if any(trail_idx)
                        plot(eye_x(trail_idx), eye_y(trail_idx), '-', ...
                            'Color', [0.8 0.1 0.1], ...
                            'LineWidth', 1.2);
                    end

                end

                %% Current gaze

                [~, idx_eye] = min(abs(eye_t_ms - t_ms));
                eye_sample_error_ms = abs(eye_t_ms(idx_eye) - t_ms);

                if eye_sample_error_ms <= MAX_EYE_SAMPLE_ERROR_MS
                    gaze_x = eye_x(idx_eye);
                    gaze_y = eye_y(idx_eye);
                else
                    gaze_x = NaN;
                    gaze_y = NaN;
                end

                draw_gaze = ~isnan(gaze_x) && ~isnan(gaze_y);

                if SKIP_GAZE_OUTSIDE_AXIS
                    draw_gaze = draw_gaze && ...
                        gaze_x >= AX_LIMS(1) && gaze_x <= AX_LIMS(2) && ...
                        gaze_y >= AX_LIMS(3) && gaze_y <= AX_LIMS(4);
                end

                if draw_gaze

                    plot(gaze_x, gaze_y, 'o', ...
                        'MarkerSize', GAZE_MARKER_SIZE, ...
                        'MarkerFaceColor', 'none', ...
                        'MarkerEdgeColor', [0.9 0 0], ...
                        'LineWidth', 3);

                    plot(gaze_x, gaze_y, '.', ...
                        'MarkerSize', 18, ...
                        'Color', [0.9 0 0]);

                end

                %% Timeline

                y_timeline = -13.5;
                x0_timeline = -12;
                x1_timeline = 12;

                map_time_to_x = @(t) x0_timeline + ((t - movie_start_ms) ./ max(movie_end_ms - movie_start_ms, eps)) .* (x1_timeline - x0_timeline);

                plot([x0_timeline x1_timeline], [y_timeline y_timeline], 'k-', 'LineWidth', 2);

                cur_x = map_time_to_x(t_ms);
                plot(cur_x, y_timeline, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');

                event_times = [flash1_ms, flash2_ms, flash3_ms];
                event_labels = {'F1', 'F2', 'F3'};

                for ev = 1:3

                    if event_times(ev) >= movie_start_ms && event_times(ev) <= movie_end_ms

                        xx = map_time_to_x(event_times(ev));

                        plot([xx xx], [y_timeline-0.4 y_timeline+0.4], 'r-', 'LineWidth', 2);

                        text(xx, y_timeline-0.8, event_labels{ev}, ...
                            'HorizontalAlignment', 'center', ...
                            'FontSize', 10, ...
                            'Color', 'r');

                    end

                end

                if ~isnan(fix_start_ms) && fix_start_ms >= movie_start_ms && fix_start_ms <= movie_end_ms

                    xx = map_time_to_x(fix_start_ms);

                    plot([xx xx], [y_timeline-0.25 y_timeline+0.25], ...
                        'Color', [0 0.4 1], ...
                        'LineWidth', 1.5);

                    text(xx, y_timeline+0.65, 'fix on', ...
                        'HorizontalAlignment', 'center', ...
                        'FontSize', 8, ...
                        'Color', [0 0.4 1]);

                end

                if ~isnan(fixation_off_ms) && fixation_off_ms >= movie_start_ms && fixation_off_ms <= movie_end_ms

                    xx = map_time_to_x(fixation_off_ms);

                    plot([xx xx], [y_timeline-0.35 y_timeline+0.35], ...
                        'Color', [0.6 0 0.8], ...
                        'LineWidth', 1.8);

                    text(xx, y_timeline+0.95, 'fix off', ...
                        'HorizontalAlignment', 'center', ...
                        'FontSize', 8, ...
                        'Color', [0.6 0 0.8]);

                end

                if ~isnan(answer_ms) && answer_ms >= movie_start_ms && answer_ms <= movie_end_ms

                    xx = map_time_to_x(answer_ms);

                    plot([xx xx], [y_timeline-0.25 y_timeline+0.25], ...
                        'Color', [0 0.6 0], ...
                        'LineWidth', 1.5);

                    text(xx, y_timeline+0.65, 'ans', ...
                        'HorizontalAlignment', 'center', ...
                        'FontSize', 8, ...
                        'Color', [0 0.6 0]);

                end

                text(x0_timeline, y_timeline+1.1, sprintf('%.0f ms', movie_start_ms), ...
                    'HorizontalAlignment', 'center', ...
                    'FontSize', 8);

                text(x1_timeline, y_timeline+1.1, sprintf('%.0f ms', movie_end_ms), ...
                    'HorizontalAlignment', 'center', ...
                    'FontSize', 8);

                text(0, y_timeline+1.1, sprintf('current = %.0f ms', t_ms), ...
                    'HorizontalAlignment', 'center', ...
                    'FontSize', 10, ...
                    'FontWeight', 'bold');

                %% Save frame

                drawnow;
                frame = getframe(fig);
                writeVideo(v, frame);

            end

            close(v);
            close(fig);

            fprintf('Saved movie: %s\n', out_file);

            if ~isnan(trial_end_ms)
                fprintf('  Trial end relative to geo_present: %.3f ms\n', trial_end_ms);
            end

            status = "saved";

            qc_movie_list(end+1,:) = { ...
                string(MONKEY), ...
                string(session_name), ...
                string(eye_file), ...
                string(behavior_file), ...
                string(out_file), ...
                selected_trial_id_for_table, ...
                n_valid_trials_for_table, ...
                string(status)};

        catch ME

            warning('Error on %s %s: %s', MONKEY, session_name, ME.message);

            try
                if exist('v', 'var')
                    close(v);
                end
            catch
            end

            try
                if exist('fig', 'var') && isvalid(fig)
                    close(fig);
                end
            catch
            end

            status = "error_" + string(ME.message);

            qc_movie_list(end+1,:) = { ...
                string(MONKEY), ...
                string(session_name), ...
                string(eye_file), ...
                string(behavior_file), ...
                string(out_file), ...
                selected_trial_id_for_table, ...
                n_valid_trials_for_table, ...
                string(status)};

        end

    end

end

%% Save list

list_file = fullfile(out_dir, 'eye_qc_movie_list.csv');

writetable(qc_movie_list, list_file);

fprintf('\n============================================================\n');
fprintf('DONE\n');
fprintf('Movies saved in:\n%s\n', out_dir);
fprintf('QC list saved as:\n%s\n', list_file);
fprintf('============================================================\n');

disp(qc_movie_list);