function [alt1_c,alt2_c,alt3_c,alt4_c,correct] = Postdictive_Model(h1,h2,h3,h4,h5,h6,v,One,Two,ntrials,thresh1,scalar,truth_1,truth_2,truth_3,truth_4,off_noise,One_noise,Two_noise)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
alt1_c = zeros(1,ntrials);
alt2_c = zeros(1,ntrials);
alt3_c = zeros(1,ntrials);
alt4_c = zeros(1,ntrials);
correct = zeros(1,ntrials);
tmp1 = zeros(1,ntrials);
tmp2 = zeros(1,ntrials);


for i = 1:ntrials  
 


if h1(i) ~= h4(i)
    i_h1 = D_likelihood(scalar,0,-1,One(i),h1(i),v(i));
    i_h2 = D_likelihood(scalar,0,1,One(i),h4(i),v(i));
else
    i_h1 = D_likelihood(scalar,0,-1,One(i),h1(i),v(i));
    i_h2 = D_likelihood(scalar,0,1,One(i),h4(i),v(i));
    
    rn = rand();
    if rn >= 0.5
       i_h1 = i_h1 + eps;
       i_h2 = i_h2;
    else
       i_h1 = i_h1;
       i_h2 = i_h2 + eps;
    end     
end
  

alt1_c(i) = D_likelihood(scalar,0,-1,Two(i),h2(i),v(i));
alt2_c(i)=  D_likelihood(scalar,0,1,Two(i),h3(i),v(i));
alt3_c(i) = D_likelihood(scalar,0,-1,Two(i),h5(i),v(i));
alt4_c(i) =  D_likelihood(scalar,0,1,Two(i),h6(i),v(i));

if i_h1*(alt1_c(i) + alt2_c(i)) > i_h2*(alt3_c(i) + alt4_c(i));
    
        alt3_c(i)=  0;
        alt4_c(i)=  0;    
   
else    

        alt1_c(i)=  0;
        alt2_c(i)=  0;
        
end

if length(unique([alt1_c(i), alt2_c(i), alt3_c(i), alt4_c(i)])) < 2
    rn = rand();
    if rn >= 0.75
        alt1_c(i) = 1;
        alt2_c(i) = 0;
        alt3_c(i) = 0;
        alt4_c(i) = 0;
    elseif rn < 0.75 && rn >= 0.5
        alt1_c(i) = 0;
        alt2_c(i) = 1;
        alt3_c(i) = 0;
        alt4_c(i) = 0;
    elseif rn < 0.5 && rn >= 0.25
        alt1_c(i) = 0;
        alt2_c(i) = 0;
        alt3_c(i) = 1;
        alt4_c(i) = 0;
    elseif rn < 0.25 
        alt1_c(i) = 0;
        alt2_c(i) = 0;
        alt3_c(i) = 0;
        alt4_c(i) = 1;
    end
elseif alt1_c(i) == alt2_c(i) && max ([alt1_c(i), alt2_c(i), alt3_c(i), alt4_c(i)]) == alt1_c(i)
    rn = rand();
    if rn >= 0.5
        alt1_c(i) = 1;
        alt2_c(i) = 0;
        alt3_c(i) = 0;
        alt4_c(i) = 0;
    else
        alt1_c(i) = 0;
        alt2_c(i) = 1;
        alt3_c(i) = 0;
        alt4_c(i) = 0;
    end      
    
elseif alt1_c(i) == alt3_c(i) && max ([alt1_c(i), alt2_c(i), alt3_c(i), alt4_c(i)]) == alt1_c(i)
    rn = rand();
    if rn >= 0.5
        alt1_c(i) = 1;
        alt2_c(i) = 0;
        alt3_c(i) = 0;
        alt4_c(i) = 0;
    else
        alt1_c(i) = 0;
        alt2_c(i) = 0;
        alt3_c(i) = 1;
        alt4_c(i) = 0;
    end  
    
elseif alt1_c(i) == alt4_c(i) && max ([alt1_c(i), alt2_c(i), alt3_c(i), alt4_c(i)]) == alt1_c(i)
    rn = rand();
    if rn >= 0.5
        alt1_c(i) = 1;
        alt2_c(i) = 0;
        alt3_c(i) = 0;
        alt4_c(i) = 0;
    else
        alt1_c(i) = 0;
        alt2_c(i) = 0;
        alt3_c(i) = 0;
        alt4_c(i) = 1;
    end  
    
elseif alt2_c(i) == alt3_c(i) && max ([alt1_c(i), alt2_c(i), alt3_c(i), alt4_c(i)]) == alt2_c(i)
    rn = rand();
    if rn >= 0.5
        alt1_c(i) = 0;
        alt2_c(i) = 1;
        alt3_c(i) = 0;
        alt4_c(i) = 0;
    else
        alt1_c(i) = 0;
        alt2_c(i) = 0;
        alt3_c(i) = 1;
        alt4_c(i) = 0;
    end  
    
elseif alt2_c(i) == alt4_c(i) && max ([alt1_c(i), alt2_c(i), alt3_c(i), alt4_c(i)]) == alt2_c(i)
    rn = rand();
    if rn >= 0.5
        alt1_c(i) = 0;
        alt2_c(i) = 1;
        alt3_c(i) = 0;
        alt4_c(i) = 0;
    else
        alt1_c(i) = 0;
        alt2_c(i) = 0;
        alt3_c(i) = 0;
        alt4_c(i) = 1;
    end
    
    
elseif alt3_c(i) == alt4_c(i) && max ([alt1_c(i), alt2_c(i), alt3_c(i), alt4_c(i)]) == alt3_c(i)
    rn = rand();
    if rn >= 0.5
        alt1_c(i) = 0;
        alt2_c(i) = 0;
        alt3_c(i) = 1;
        alt4_c(i) = 0;
    else
        alt1_c(i) = 0;
        alt2_c(i) = 0;
        alt3_c(i) = 0;
        alt4_c(i) = 1;
    end  
    
elseif  max ([alt1_c(i), alt2_c(i), alt3_c(i), alt4_c(i)]) == alt1_c(i)
    alt1_c(i) = 1;
    alt2_c(i) = 0;
    alt3_c(i) = 0;
    alt4_c(i) = 0;
elseif  max ([alt1_c(i), alt2_c(i), alt3_c(i), alt4_c(i)]) == alt2_c(i)
    alt1_c(i) = 0;
    alt2_c(i) = 1;
    alt3_c(i) = 0;
    alt4_c(i) = 0;
elseif  max ([alt1_c(i), alt2_c(i), alt3_c(i), alt4_c(i)]) == alt3_c(i)
    alt1_c(i) = 0;
    alt2_c(i) = 0;
    alt3_c(i) = 1;
    alt4_c(i) = 0;
elseif  max ([alt1_c(i), alt2_c(i), alt3_c(i), alt4_c(i)]) == alt4_c(i)
    alt1_c(i) = 0;
    alt2_c(i) = 0;
    alt3_c(i) = 0;
    alt4_c(i) = 1;
else
    display (' results do not satisfy any criteria')
end 

if isequal([alt1_c(i) alt2_c(i) alt3_c(i) alt4_c(i)], [truth_1(i),truth_2(i),truth_3(i),truth_4(i)])
    correct(i) = 1;
end

end


end
