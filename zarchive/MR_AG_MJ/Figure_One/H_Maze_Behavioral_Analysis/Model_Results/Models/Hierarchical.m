function [alt1_h,alt2_h,alt3_h,alt4_h,correct] = Hierarchical(h1,h2,h3,h4,h5,h6,v,One,Two,ntrials,scalar,truth_1,truth_2,truth_3,truth_4)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

alt1_h = zeros(1,ntrials);
alt2_h = zeros(1,ntrials);
alt3_h = zeros(1,ntrials);
alt4_h = zeros(1,ntrials);
correct = zeros(1,ntrials);

for i = 1:ntrials
    
if h1(i) ~= h4(i)
    i_h1 = D_likelihood(scalar,0,-1,One(i),h1(i),v(i));
    i_h2 = D_likelihood(scalar,0,1,One(i),h4(i),v(i));
else
   rn = rand();
    if rn >= 0.5
       i_h1 = 1;
       i_h2 = 0;
    else
       i_h1 = 0;
       i_h2 = 1;
    end      
end
  
if i_h1 > i_h2;
    
 
        alt1_h(i) = D_likelihood(scalar,0,-1,Two(i),h2(i),v(i));
        alt2_h(i)=  D_likelihood(scalar,0,1,Two(i),h3(i),v(i));
        alt3_h(i)=  0;
        alt4_h(i)=  0;     
else
      
        alt3_h(i) = D_likelihood(scalar,0,-1,Two(i),h5(i),v(i));
        alt4_h(i)=  D_likelihood(scalar,0,1,Two(i),h6(i),v(i));
        alt1_h(i)=  0;
        alt2_h(i)=  0;
 
end

if length(unique([alt1_h(i), alt2_h(i), alt3_h(i), alt4_h(i)])) < 2
    rn = rand();
    if rn >= 0.75
        alt1_h(i) = 1;
        alt2_h(i) = 0;
        alt3_h(i) = 0;
        alt4_h(i) = 0;
    elseif rn < 0.75 && rn >= 0.5
        alt1_h(i) = 0;
        alt2_h(i) = 1;
        alt3_h(i) = 0;
        alt4_h(i) = 0;
    elseif rn < 0.5 && rn >= 0.25
        alt1_h(i) = 0;
        alt2_h(i) = 0;
        alt3_h(i) = 1;
        alt4_h(i) = 0;
    elseif rn < 0.25 
        alt1_h(i) = 0;
        alt2_h(i) = 0;
        alt3_h(i) = 0;
        alt4_h(i) = 1;
    end
elseif alt1_h(i) == alt2_h(i) && max ([alt1_h(i), alt2_h(i), alt3_h(i), alt4_h(i)]) == alt1_h(i)
    rn = rand();
    if rn >= 0.5
        alt1_h(i) = 1;
        alt2_h(i) = 0;
        alt3_h(i) = 0;
        alt4_h(i) = 0;
    else
        alt1_h(i) = 0;
        alt2_h(i) = 1;
        alt3_h(i) = 0;
        alt4_h(i) = 0;
    end      
    
elseif alt1_h(i) == alt3_h(i) && max ([alt1_h(i), alt2_h(i), alt3_h(i), alt4_h(i)]) == alt1_h(i)
    rn = rand();
    if rn >= 0.5
        alt1_h(i) = 1;
        alt2_h(i) = 0;
        alt3_h(i) = 0;
        alt4_h(i) = 0;
    else
        alt1_h(i) = 0;
        alt2_h(i) = 0;
        alt3_h(i) = 1;
        alt4_h(i) = 0;
    end  
    
elseif alt1_h(i) == alt4_h(i) && max ([alt1_h(i), alt2_h(i), alt3_h(i), alt4_h(i)]) == alt1_h(i)
    rn = rand();
    if rn >= 0.5
        alt1_h(i) = 1;
        alt2_h(i) = 0;
        alt3_h(i) = 0;
        alt4_h(i) = 0;
    else
        alt1_h(i) = 0;
        alt2_h(i) = 0;
        alt3_h(i) = 0;
        alt4_h(i) = 1;
    end  
    
elseif alt2_h(i) == alt3_h(i) && max ([alt1_h(i), alt2_h(i), alt3_h(i), alt4_h(i)]) == alt2_h(i)
    rn = rand();
    if rn >= 0.5
        alt1_h(i) = 0;
        alt2_h(i) = 1;
        alt3_h(i) = 0;
        alt4_h(i) = 0;
    else
        alt1_h(i) = 0;
        alt2_h(i) = 0;
        alt3_h(i) = 1;
        alt4_h(i) = 0;
    end  
    
elseif alt2_h(i) == alt4_h(i) && max ([alt1_h(i), alt2_h(i), alt3_h(i), alt4_h(i)]) == alt2_h(i)
    rn = rand();
    if rn >= 0.5
        alt1_h(i) = 0;
        alt2_h(i) = 1;
        alt3_h(i) = 0;
        alt4_h(i) = 0;
    else
        alt1_h(i) = 0;
        alt2_h(i) = 0;
        alt3_h(i) = 0;
        alt4_h(i) = 1;
    end
    
    
elseif alt3_h(i) == alt4_h(i) && max ([alt1_h(i), alt2_h(i), alt3_h(i), alt4_h(i)]) == alt3_h(i)
    rn = rand();
    if rn >= 0.5
        alt1_h(i) = 0;
        alt2_h(i) = 0;
        alt3_h(i) = 1;
        alt4_h(i) = 0;
    else
        alt1_h(i) = 0;
        alt2_h(i) = 0;
        alt3_h(i) = 0;
        alt4_h(i) = 1;
    end  
    
elseif  max ([alt1_h(i), alt2_h(i), alt3_h(i), alt4_h(i)]) == alt1_h(i)
    alt1_h(i) = 1;
    alt2_h(i) = 0;
    alt3_h(i) = 0;
    alt4_h(i) = 0;
elseif  max ([alt1_h(i), alt2_h(i), alt3_h(i), alt4_h(i)]) == alt2_h(i)
    alt1_h(i) = 0;
    alt2_h(i) = 1;
    alt3_h(i) = 0;
    alt4_h(i) = 0;
elseif  max ([alt1_h(i), alt2_h(i), alt3_h(i), alt4_h(i)]) == alt3_h(i)
    alt1_h(i) = 0;
    alt2_h(i) = 0;
    alt3_h(i) = 1;
    alt4_h(i) = 0;
elseif  max ([alt1_h(i), alt2_h(i), alt3_h(i), alt4_h(i)]) == alt4_h(i)
    alt1_h(i) = 0;
    alt2_h(i) = 0;
    alt3_h(i) = 0;
    alt4_h(i) = 1;
else
    display (' results do not satisfy any criteria');
end

if isequal([alt1_h(i) alt2_h(i) alt3_h(i) alt4_h(i)], [truth_1(i),truth_2(i),truth_3(i),truth_4(i)])
    correct(i) = 1;
end


end

end

