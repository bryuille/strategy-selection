function [alt1,alt2,alt3,alt4,correct] = Revision_X_LR_Max_Model_Physiology( ...
        h1,h2,h3,h4,h5,h6,one,two,n_trials,threshold,scalar, ...
        truth_1,truth_2,truth_3,truth_4,noise_slope,noise_offset,two_noise)
% Figure 5 physiology-input implementation of the retained Figure 1
% hierarchical-with-revision policy. Arm lengths and measurements are
% supplied in seconds.
assert(numel(one) == n_trials && numel(two) == n_trials && ...
    numel(two_noise) == n_trials, 'Revision measurement/trial-count mismatch.');
outcomes = zeros(n_trials, 4);
correct = zeros(1, n_trials);
noise_scalar = scalar * noise_slope;

for trial = 1:n_trials
    left_first = D_likelihood_Physiology(scalar,0,one(trial),h1(trial));
    right_first = D_likelihood_Physiology(scalar,0,one(trial),h4(trial));
    if left_first == right_first
        if rand >= 0.5, left_first = left_first + eps;
        else, right_first = right_first + eps;
        end
    end

    scores = zeros(1,4);
    if left_first > right_first
        scores(1) = D_likelihood_Physiology(scalar,0,two(trial),h2(trial));
        scores(2) = D_likelihood_Physiology(scalar,0,two(trial),h3(trial));
        if left_first * max(scores(1:2)) < threshold
            scores(:) = 0;
            scores(3) = D_likelihood_Physiology(noise_scalar,noise_offset, ...
                two_noise(trial),h5(trial));
            scores(4) = D_likelihood_Physiology(noise_scalar,noise_offset, ...
                two_noise(trial),h6(trial));
        end
    else
        scores(3) = D_likelihood_Physiology(scalar,0,two(trial),h5(trial));
        scores(4) = D_likelihood_Physiology(scalar,0,two(trial),h6(trial));
        if right_first * max(scores(3:4)) < threshold
            scores(:) = 0;
            scores(1) = D_likelihood_Physiology(noise_scalar,noise_offset, ...
                two_noise(trial),h2(trial));
            scores(2) = D_likelihood_Physiology(noise_scalar,noise_offset, ...
                two_noise(trial),h3(trial));
        end
    end
    tied = find(scores == max(scores));
    selected = tied(randi(numel(tied)));
    outcomes(trial, selected) = 1;
    correct(trial) = isequal(outcomes(trial, :), ...
        [truth_1(trial),truth_2(trial),truth_3(trial),truth_4(trial)]);
end
alt1 = outcomes(:,1)'; alt2 = outcomes(:,2)';
alt3 = outcomes(:,3)'; alt4 = outcomes(:,4)';
end
