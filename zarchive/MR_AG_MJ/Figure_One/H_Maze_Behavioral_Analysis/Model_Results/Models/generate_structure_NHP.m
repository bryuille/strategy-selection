function [h1,h2,h3,h4,h5,h6,d1,d2,v,One,Two,truet1,truet2,truth_1,truth_2,truth_3,truth_4, One_noise, Two_noise, Total_noise] = generate_structure_NHP(h1,h2,h3,h4,h5,h6,v,d1,d2,ntrials,min_t,scalar,off_noise1,off_noise2, scal_noise1, scal_noise2)


One = zeros(1,ntrials);
Two = zeros(1,ntrials);
One_noise = zeros(1,ntrials);
Two_noise = zeros(1,ntrials);
Total_noise = zeros(1,ntrials);

truet1 = zeros(1,ntrials);
truet2 = zeros(1,ntrials);

truth_1 = zeros(1,ntrials);
truth_2 = zeros(1,ntrials);
truth_3 = zeros(1,ntrials);
truth_4 = zeros(1,ntrials);


% should sample tm from scalar dist., decision with scalar variability 
for i = 1:ntrials
    if d1(i) == 1
   
       One(i) = max(normrnd(h4(i)./v(i),scalar*(h4(i)./v(i))), min_t);
       One_noise(i) = max(normrnd(h4(i)./v(i),scal_noise1 + off_noise1*scalar*(h4(i)./v(i))), min_t);

       truet1(i) = h4(i)./v(i);
       
       if  d2(i) == -1
           
            Two(i) = max(normrnd(h6(i)./v(i),scalar*(h6(i)./v(i))),min_t);
            Two_noise(i) = max(normrnd(h6(i)./v(i),scal_noise2 + off_noise2*scalar*(h6(i)./v(i))),min_t);
            Total_noise(i) = max(normrnd((h4(i) + h6(i))./v(i), scal_noise2 + off_noise2*scalar*((h4(i) + h6(i))./v(i))),min_t);          
            truet2(i) = h6(i)./v(i);
            truth_1(i) = 0;
            truth_2(i) = 0;
            truth_3(i) = 0;
            truth_4(i) = 1;
            
       elseif d2(i) == 1
     
            Two(i) = max(normrnd(h5(i)./v(i),scalar*(h5(i)./v(i))), min_t);
            Two_noise(i) = max(normrnd(h5(i)./v(i), scal_noise2 + off_noise2*scalar*(h5(i)./v(i))), min_t);
            Total_noise(i) = max(normrnd((h4(i) + h5(i))./v(i),scal_noise2 + off_noise2*scalar*((h4(i) + h5(i))./v(i))),min_t);
            truet2(i) = h5(i)./v(i);  
            truth_1(i) = 0;
            truth_2(i) = 0;
            truth_3(i) = 1;
            truth_4(i) = 0;
       end

    elseif  d1(i) == -1
       One(i) = max(normrnd(h1(i)./v(i),scalar*(h1(i)./v(i))),min_t);
       One_noise(i) = max(normrnd(h1(i)./v(i),scal_noise1 + off_noise1*scalar*(h1(i)./v(i))),min_t);

       truet1(i) = h1(i)./v(i);
       
       if d2(i) == 1
   
            Two(i) = max(normrnd(h2(i)./v(i),scalar*(h2(i)./v(i))), min_t);
            Two_noise(i) = max(normrnd(h2(i)./v(i),scal_noise2 + off_noise2*scalar*(h2(i)./v(i))), min_t);
            Total_noise(i) = max(normrnd((h1(i) + h2(i))./v(i),scal_noise2 + off_noise2*scalar*((h1(i) + h2(i))./v(i))),min_t);
            truet2(i) = h2(i)./v(i);
            truth_1(i) = 1;
            truth_2(i) = 0;
            truth_3(i) = 0;
            truth_4(i) = 0;
                
                
       elseif  d2(i) == -1
            
           Two(i) = max(normrnd(h3(i)./v(i),scalar*(h3(i)./v(i))), min_t);
           Two_noise(i) = max(normrnd(h3(i)./v(i),scal_noise2 + off_noise2*scalar*(h3(i)./v(i))), min_t);
           Total_noise(i) = max(normrnd((h1(i) + h3(i))./v(i),scal_noise2 + off_noise2*scalar*((h1(i) + h3(i))./v(i))),min_t);


            truet2(i) = h3(i)./v(i);        
            truth_1(i) = 0;
            truth_2(i) = 1;
            truth_3(i) = 0;
            truth_4(i) = 0;

        end
     end
   end
end


