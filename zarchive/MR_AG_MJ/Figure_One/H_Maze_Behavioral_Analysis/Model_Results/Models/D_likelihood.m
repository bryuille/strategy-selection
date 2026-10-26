function [outputArg1 ] = D_likelihood(scalar,offset, alt,One,h1,v)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% temp = normcdf(5./v, truet, scalar*truet);

outputArg1 = normpdf(One,h1./v, (h1./v)*scalar + offset);

% Calculate p(tm|ts,wm)
% x = ((h1./v) - One).^2;
% wmts2 = ((h1./v)*scalar).^2;
% outputArg1  = 1./sqrt(2*pi*wmts2).*exp(-0.5*x./wmts2);
% 

 
end


