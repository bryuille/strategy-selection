function [fval,p_left_up,p_left_down, p_right_up,  p_right_down , Input_test, test_indices] = model_full_optimal(Input, trainIdx, testIdx)
%% Simulate Model
function [p_left_up,p_left_down, p_right_up,  p_right_down] = Pr_Left_ts(Input, parameters);

h1_hum = Input.h1_hum ;
h2_hum =  Input.h2_hum;
h3_hum = Input.h3_hum ;
h4_hum = Input.h4_hum;
h5_hum = Input.h5_hum ;
h6_hum =  Input.h6_hum;
v_hum = Input.v_hum ;
d1_hum = Input.d1_hum;
d2_hum = Input.d2_hum ;
min_t = Input.min_t;
scalar = Input.scalar;
samples = 1000; 

    parfor iTrial = 1: length(h1_hum)
        
        alt1_all = [];
        alt2_all = [];
        alt3_all = [];
        alt4_all = [];
        
        for k = 1:samples
                        
            [h1,h2,h3,h4,h5,h6,d1,d2,v,One,Two,truet1,truet2,truth_1,truth_2,truth_3,truth_4,One_noise,Two_noise, Total_noise] = generate_structure_NHP(h1_hum(iTrial), h2_hum(iTrial), h3_hum(iTrial),h4_hum(iTrial),h5_hum(iTrial),h6_hum(iTrial),  v_hum(iTrial), d1_hum(iTrial), d2_hum(iTrial) , length(h1_hum(iTrial)),min_t,scalar, 1,1,0,0);
            [alt1,alt2,alt3,alt4,correctnh] = Optimal_Model(h1,h2,h3,h4,h5,h6,v,One,Two,length(h1_hum(iTrial)),scalar,truth_1,truth_2,truth_3,truth_4);

            alt1_all = horzcat(alt1_all, alt1);
            alt2_all = horzcat(alt2_all, alt2);
            alt3_all = horzcat(alt3_all, alt3);
            alt4_all = horzcat(alt4_all, alt4);

        end
        
        
        % Initialize key and key codes
        key = zeros(1, length(alt1_all));
        key(alt1_all == 1) = 1;
        key(alt2_all == 1) = 2;
        key(alt3_all == 1) = 3;
        key(alt4_all == 1) = 4;
        
        key_code_1 = zeros(1, length(alt1_all));
        key_code_2 = zeros(1, length(alt1_all));
        key_code_3 = zeros(1, length(alt1_all));
        key_code_4 = zeros(1, length(alt1_all));
        
        % Iterate through samples and assign key codes directly based on key
        for i = 1:samples
            if key(i) == 1
                key_code_1(i) = 1;
            elseif key(i) == 2
                key_code_2(i) = 1;
            elseif key(i) == 3
                key_code_3(i) = 1;
            elseif key(i) == 4
                key_code_4(i) = 1;
            end
        end
        
        % Calculate the mean values for each key code
        truth1_hum = mean(key_code_1);
        truth2_hum = mean(key_code_2);
        truth3_hum = mean(key_code_3);
        truth4_hum = mean(key_code_4);
        
        % Assign probabilities to trials
        p_left_up(iTrial) = truth1_hum;
        p_left_down(iTrial) = truth2_hum;
        p_right_up(iTrial) = truth3_hum;
        p_right_down(iTrial) = truth4_hum;

    
    end
        
end


%% negative log-likelihood 
function [logLikelihoodValue,p_left_up,p_left_down, p_right_up,  p_right_down ] = logLikelihood_Of_BernouliDist_p_left_ts(Input, parameters);
% Model parameters
[p_left_up,p_left_down, p_right_up,  p_right_down] = Pr_Left_ts(Input, parameters);
left_choice = Input.left_choice; % 0:Right, 1: Left
up_choice = Input.up_choice;

% logLikelihoodValue = sqrt(sum((RT(find(TF==1)) - RT_Model(find(TF==1))).^2)/length(RT(find(TF==1)))) -sum(Anti.*log(p_Anti_td_C+0.0001) + (1-Anti).*log(1 - p_Anti_td_C+0.0001));
logLikelihoodValue = -sum( left_choice.* up_choice.*log(p_left_up+eps) + (left_choice).*(1-up_choice).*log(p_left_down+eps) + (1-left_choice).*(up_choice).*log(p_right_up+eps) + (1-left_choice).*(1- up_choice).*log(p_right_down+eps) );
% logLikelihoodValue =  -sum(Anti.*log(p_Anti_td_C+0.0001) + (1-Anti).*log(1 - p_Anti_td_C+0.0001));
end

%% optimization

% Setting the initialized values of optimizer
% options = optimset('fminsearch');
% options.Display = 'iter';
% options.Iter = 1000000;
% options.TolFun = 1e-6;
% options.TolX = 1e-6;


if nargin < 3
    trainIdx = randperm(numel(Input.h1_hum), round(numel(Input.h1_hum)*.7));
    testIdx = find(~ismember(1:numel(Input.h1_hum), trainIdx));
end
test_indices = testIdx;


Input_test.h1_hum = Input.h1_hum(testIdx);
Input_test.h2_hum = Input.h2_hum(testIdx);
Input_test.h3_hum = Input.h3_hum(testIdx);
Input_test.h4_hum = Input.h4_hum(testIdx);
Input_test.h5_hum = Input.h5_hum(testIdx);
Input_test.h6_hum = Input.h6_hum(testIdx);
Input_test.v_hum = Input.v_hum(testIdx);
Input_test.d1_hum = Input.d1_hum(testIdx);
Input_test.d2_hum = Input.d2_hum(testIdx);
Input_test.min_t = Input.min_t;


Input_test.left_choice = Input.left_choice(testIdx) ;
Input_test.up_choice =  Input.up_choice (testIdx);
Input_test.scalar =  Input.scalar; 
Input_test.eucnhpx = Input.eucnhpx;
Input_test.eucnhpy = Input.eucnhpy;
Input_test.eucnhpz = Input.eucnhpz;


[fval,p_left_up,p_left_down, p_right_up,  p_right_down ] = logLikelihood_Of_BernouliDist_p_left_ts(Input_test, 1);


end
