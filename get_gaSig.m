function [X maxvalue]=get_gaSig(RefID,startParm,nvar, valuemin, valuemax, workers)

%% all get_gaFunc are the same, only difference is the function being evaluated


display('Starting GA');

format long

if nargin < 6
    
   workers = 4; 
    
end


%% to read the data it will create global noteletCache: remember to clear it at the end of the file


display('initialize matlabpool');


if workers>1
  	if matlabpool('size') ~= workers && matlabpool('size') > 0
	      matlabpool close;
          matlabpool('local', workers);

    elseif matlabpool('size')==0
          
          matlabpool('local', workers);

    end
end
     

%------------------------        parameters        ------------------------
% befor using this function you must specified your function in fun00.m
% file in current directory and then set the parameters

var = nvar;


n=40 ;            % Number of population

m0=20;            % Number of generations that max value remains constant
                  %   (use for termination criteria)

                  
nMaxGen   =150;%280;                %maximum number of generations/default 1e4                  
nMinGen   = 90;



%% hack
m0= 10;            % Number of generations that max value remains constant
nMaxGen   = 30;%280;                %maximum number of generations/default 1e4                  
nMinGen   = 15;
%%
% %% hack
 m0= 8;            % Number of generations that max value remains constant
nMaxGen   = 35;%280;                %maximum number of generations/default 1e4                  
nMinGen   = 15;


nmutationG=1+floor(.025*n);     %number of mutation children(Gaussian)
nmutationR=1+floor(.5*n);     % initially was 15%; number of mutation children(random)



nelit=1+floor(.05*n);           %number of elitism children: about 5% population



valueminMat= repmat(valuemin,n,1);
valuemaxMat= repmat(valuemax,n,1);



%-------------------------------------------------------------------------
nmutation=  nmutationG+nmutationR;
sigma    = (valuemax-valuemin)/5;    %Parameter that related to Gaussian
sigmaMat = (valuemaxMat-valueminMat)/5;

%   function and used in mutation step
max1=zeros(nelit,var);
parent=zeros(n,var);


p = valueminMat+rand(n,var).*(valuemaxMat-valueminMat);

 if length(startParm)==var
    
   p(end,:) = startParm;
 
 end 
    


m=m0;
maxvalue=ones(m,1)*-1e10;
maxvalue(m)=-1e5;
g  = 0;
gs = 0;
meanvalue(m)=0;
ii=0;
sigmaDecay = 1.05 ; %its inverse is the amount sigma is reduced 
                    %at each iteration 

y          = zeros(n,1);                    
stop       = false;                   
                    
%-------------   ****    termination criteria   ****-------------
while (m<nMaxGen && abs(maxvalue(m)-maxvalue(m-(m0-1)))>0.001) ||  m <nMinGen && ~stop

    
%      abs(maxvalue(m)-maxvalue(m-(m0-1)))>0.001*maxvalue(m) &...
%          (abs(maxvalue(m))>1e-10 & abs(maxvalue(m-(m0-1)))>1e-10)...
%          & m<nMaxGen & abs(maxvalue(m)-meanvalue(m))>1e-5 | m<20
    
    
    ii=ii+1;
    display(ii)
    
    
%------------------- reducing the sigma value
    
     sigmaDecay_gs = sigmaDecay^(1+gs);

     sigma         = sigma./sigmaDecay_gs;% reducing the sigma value
     sigmaMat      = sigmaMat/sigmaDecay_gs;

    gs=gs+1;
    
    if gs>10 %&& nmutationR>0
        gs=0;
    end

    
    %-------------   ****    function evaluation   ****-------------
    tic  
   
  parfor i=1:n
      
    RefData = readsharedata(RefID);
    y(i)    = fun_ga_aggJava(p(i,:), RefData);

  end
    toc
    s         = sort(y,'descend');
    maxvalue1 = s(1:nelit);
    
  
   
%------------------------ select elite
   for k=1:nelit    
      Ind      = (y == maxvalue1(k));
      tMax     =  p(Ind,:);    
      max1(k,:)=  tMax(1,:);   
   end   
  
% % % % %      IndBest  = find(y == maxvalue1(1));
% % % % %      pBest    = p(IndBest(1),:);
% % % % %         


%%

%% perturbate elite

nPerturbation      = (floor(n/4)+1);
sigmaPerturbation  = .1*std(p);

phi                =  1-2*rand(nPerturbation,var);
z                  =  erfinv(phi)*(2^0.5);
perturbation       =  bsxfun(@times, z, sigmaPerturbation);
yElite             =  zeros(nPerturbation,nelit);    

for k =1:nelit

    pElite         =  bsxfun(@plus, max1(k,:), perturbation);
    pElite         =  bsxfun(@max, pElite,valueminMat(1,:));
    pElite         =  bsxfun(@min, pElite,valuemaxMat(1,:));

    parfor i=1:nPerturbation

        RefData      = readsharedata(RefID);
        yElite(i,k)  = fun_ga_aggJava(pElite(i,:), RefData);

    end

 
end

[maxElite indBest]   = max(mean(yElite));
pBest                = max1(indBest,:);
maxvalue1(1)         = maxvalue1(indBest); 

stop                 = (ii> 10) && (maxElite <.1); 
 %-------------   ****   Selection: Tournament   ****-------------
       

    %% better to use randsample(n,n,1) ?

    Ind                           = 1+floor(rand(n,2)*n);
    mask                          = ( y(Ind(:,1))>y(Ind(:,2)) );
    Ind1                          = Ind(mask,1);
    Ind2                          = Ind(~mask,2);
    
    parent(1:length(Ind1),:)        = p(Ind1,:);
    parent((length(Ind1)+1):end,:)  = p(Ind2,:);
   
    p                             = zeros(n,var);   
 
    
    
    %-------------   ****    regeneration   ****-------------
    
    %------------ ****    crossover  ***------------
  
 nXover      = (n-nmutation-nelit);
 t           =  rand(nXover,1)*1.5-0.25;
 t           =  repmat(t,1,var);
 Ind         = 1+floor(rand(nXover,2)*n);
 p(1:nXover,:)  = t.*parent(Ind(:,1),:)+(1-t).*parent(Ind(:,2),:);

 
        %-------------   ****    elitism   ****-------------
%  
%  if perturbedElitism       
% 
%     parfor i=1:n
%         
%     phi         =  1-2*rand(nInd,var);
%     z           =  erfinv(phi)*(2^0.5);
%         =  z.*sigmaMat(Ind,:)+parent(Ind,:);
%     perturbed    = .05*max1(i,:).*    
%         
%     RefData = readsharedata(RefID);
%     y(i)    = fun_ga_aggJava(max1(i,:), RefData);
% 
%   end
% 
%      
%      
%  else
 p((nXover+1):(n-nmutation),:) = max1; 
%  end
 
 %-------------   ****    mutation   ****------------  
  
    Ind  = [(n-nmutation+1):1:(n-nmutationR)];
    nInd = length(Ind);    
        
    phi         =  1-2*rand(nInd,var);
    z           =  erfinv(phi)*(2^0.5);
    p(Ind,:)    =  z.*sigmaMat(Ind,:)+parent(Ind,:);
               
    Ind      = [(n-nmutationR)+1:1:n];
    nInd     = length(Ind);
      
    p(Ind,:) = valueminMat(Ind,:)...
               +rand(nInd,var).*(valuemaxMat(Ind,:)-valueminMat(Ind,:));
           
           
   %------------------impose constraints-----------      
    p=max(p,valueminMat);
    p=min(p,valuemaxMat);
    
    
    m=m+1;
    max1;
    maxvalue(m)=maxvalue1(1);
    maxvalue00(m-m0)=maxvalue1(1);
    mean00(m-m0)=sum(s)/n;
    meanvalue(m)=mean00(m-m0);
    
 display([ maxvalue00(end)  mean00(end)])      

  display('Best Parameter')
  display( pBest(1,:))
end

if matlabpool('size') > 0

    matlabpool close;
    
end

clear global % noteletCache ; % these are the data created using the readsharedata in parfor function


disp('**************************************')
num_of_fun_evaluation=n*m
max_point_GA=max1(1,:)
maxvalue_GA=maxvalue00(m-m0)
disp('**************************************')



figure(1)
title('Performance of GA(best value)','color','b')
xlabel('number of generations')
ylabel('max value of best solution')
hold on
plot(maxvalue00)
hold on


X        = max_point_GA;
maxvalue = maxvalue00(m-m0);
toc
