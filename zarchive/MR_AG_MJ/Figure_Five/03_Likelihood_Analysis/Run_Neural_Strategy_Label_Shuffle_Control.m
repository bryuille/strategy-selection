function Run_Neural_Strategy_Label_Shuffle_Control(n_samples,n_label_shuffles)
% Run the neural-label-versus-opposite analysis plus a within-session label
% permutation control. Canonical and random-geometry labels are shuffled
% together within each session; choices and H/S model likelihoods remain fixed.

if nargin < 1
    n_samples=10000;
end
if nargin < 2
    n_label_shuffles=10000;
end
Run_Canonical_Neural_Strategy_vs_Trialwise_Opposite( ...
    n_samples,n_label_shuffles);
end
