function probabilities = Apply_Optimal_Lapse_Parameters_Physiology( ...
        optimal_probabilities, lapse_parameters)
% Apply each independently fitted Figure 1 lapse rate and average the
% resulting predictive probabilities (not the parameter values).
n_fits = numel(lapse_parameters);
fit_probabilities = zeros([size(optimal_probabilities), n_fits]);
for fit_idx = 1:n_fits
    lapse = lapse_parameters(fit_idx);
    assert(isfinite(lapse) && lapse >= 0 && lapse <= 1, ...
        'Invalid lapse parameter in fit %d.', fit_idx);
    fit_probabilities(:,:,fit_idx) = ...
        (1-lapse) .* optimal_probabilities + lapse .* 0.25;
end
probabilities = mean(fit_probabilities, 3);
end
