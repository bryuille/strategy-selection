function [alt1, alt2, alt3, alt4,correct] = total_time(h1,h2,h3,h4,h5,h6,v,One,Two,ntrials,scalar,truth_1,truth_2,truth_3,truth_4, off2, scal_noise2)
% Non-Hierarchical model

alt1 = zeros(1,ntrials);
alt2 = zeros(1,ntrials);
alt3 = zeros(1,ntrials);
alt4 = zeros(1,ntrials);
correct = zeros(1,ntrials);

for i = 1:ntrials
     
alt1(i) = D_likelihood(scalar*off2, scal_noise2,-1,One(i),h1(i) + h2(i),v(i));
alt2(i) = D_likelihood(scalar*off2 ,scal_noise2,-1,One(i),h1(i) + h3(i),v(i));
alt3(i) = D_likelihood(scalar*off2 , scal_noise2,1,One(i),h4(i) + h5(i),v(i));
alt4(i) = D_likelihood(scalar*off2 , scal_noise2,1,One(i),h4(i) + h6(i),v(i));
   
if length(unique([alt1(i), alt2(i), alt3(i), alt4(i)])) < 2
    rn = rand();
    if rn >= 0.75
        alt1(i) = 1;
        alt2(i) = 0;
        alt3(i) = 0;
        alt4(i) = 0;
    elseif rn < 0.75 && rn >= 0.5
        alt1(i) = 0;
        alt2(i) = 1;
        alt3(i) = 0;
        alt4(i) = 0;
    elseif rn < 0.5 && rn >= 0.25
        alt1(i) = 0;
        alt2(i) = 0;
        alt3(i) = 1;
        alt4(i) = 0;
    elseif rn < 0.25 
        alt1(i) = 0;
        alt2(i) = 0;
        alt3(i) = 0;
        alt4(i) = 1;
    end
elseif alt1(i) == alt2(i) && max ([alt1(i), alt2(i), alt3(i), alt4(i)]) == alt1(i)
    rn = rand();
    if rn >= 0.5
        alt1(i) = 1;
        alt2(i) = 0;
        alt3(i) = 0;
        alt4(i) = 0;
    else
        alt1(i) = 0;
        alt2(i) = 1;
        alt3(i) = 0;
        alt4(i) = 0;
    end      
    
elseif alt1(i) == alt3(i) && max ([alt1(i), alt2(i), alt3(i), alt4(i)]) == alt1(i)
    rn = rand();
    if rn >= 0.5
        alt1(i) = 1;
        alt2(i) = 0;
        alt3(i) = 0;
        alt4(i) = 0;
    else
        alt1(i) = 0;
        alt2(i) = 0;
        alt3(i) = 1;
        alt4(i) = 0;
    end  
    
elseif alt1(i) == alt4(i) && max ([alt1(i), alt2(i), alt3(i), alt4(i)]) == alt1(i)
    rn = rand();
    if rn >= 0.5
        alt1(i) = 1;
        alt2(i) = 0;
        alt3(i) = 0;
        alt4(i) = 0;
    else
        alt1(i) = 0;
        alt2(i) = 0;
        alt3(i) = 0;
        alt4(i) = 1;
    end  
    
elseif alt2(i) == alt3(i) && max ([alt1(i), alt2(i), alt3(i), alt4(i)]) == alt2(i)
    rn = rand();
    if rn >= 0.5
        alt1(i) = 0;
        alt2(i) = 1;
        alt3(i) = 0;
        alt4(i) = 0;
    else
        alt1(i) = 0;
        alt2(i) = 0;
        alt3(i) = 1;
        alt4(i) = 0;
    end  
    
elseif alt2(i) == alt4(i) && max ([alt1(i), alt2(i), alt3(i), alt4(i)]) == alt2(i)
    rn = rand();
    if rn >= 0.5
        alt1(i) = 0;
        alt2(i) = 1;
        alt3(i) = 0;
        alt4(i) = 0;
    else
        alt1(i) = 0;
        alt2(i) = 0;
        alt3(i) = 0;
        alt4(i) = 1;
    end
    
    
elseif alt3(i) == alt4(i) && max ([alt1(i), alt2(i), alt3(i), alt4(i)]) == alt3(i)
    rn = rand();
    if rn >= 0.5
        alt1(i) = 0;
        alt2(i) = 0;
        alt3(i) = 1;
        alt4(i) = 0;
    else
        alt1(i) = 0;
        alt2(i) = 0;
        alt3(i) = 0;
        alt4(i) = 1;
    end  
    
elseif  max ([alt1(i), alt2(i), alt3(i), alt4(i)]) == alt1(i)
    alt1(i) = 1;
    alt2(i) = 0;
    alt3(i) = 0;
    alt4(i) = 0;
elseif  max ([alt1(i), alt2(i), alt3(i), alt4(i)]) == alt2(i)
    alt1(i) = 0;
    alt2(i) = 1;
    alt3(i) = 0;
    alt4(i) = 0;
elseif  max ([alt1(i), alt2(i), alt3(i), alt4(i)]) == alt3(i)
    alt1(i) = 0;
    alt2(i) = 0;
    alt3(i) = 1;
    alt4(i) = 0;
elseif  max ([alt1(i), alt2(i), alt3(i), alt4(i)]) == alt4(i)
    alt1(i) = 0;
    alt2(i) = 0;
    alt3(i) = 0;
    alt4(i) = 1;
else
    display (' results do not satisfy any criteria');
end 


if isequal([alt1(i) alt2(i) alt3(i) alt4(i)], [truth_1(i),truth_2(i),truth_3(i),truth_4(i)])
    correct(i) = 1;
end

end
end


%D_likelihood(scalar,1,One(find(isnan(alt1))),h1(find(isnan(alt1))),h4(find(isnan(alt1))),v(find(isnan(alt1)))).*D_likelihood(scalar,1,Two(find(isnan(alt1))),h5(find(isnan(alt1))),h6(find(isnan(alt1))),v(find(isnan(alt1)))).*(normpdf(One(find(isnan(alt1))),h4(find(isnan(alt1)))./v(find(isnan(alt1))), prior_s)./normpdf(h4(find(isnan(alt1)))./v(find(isnan(alt1))),h4(find(isnan(alt1)))./v(find(isnan(alt1))), prior_s)).*(normpdf(Two(find(isnan(alt1))),h6(find(isnan(alt1)))./v(find(isnan(alt1))), prior_s)./normpdf(h6(find(isnan(alt1)))./v(find(isnan(alt1))),h6(find(isnan(alt1)))./v(find(isnan(alt1))), prior_s))