function [X maxvalue]=OLSGAparAS2(Predictors, Price, Spread, Time, Brokerage, onePip, pipette,InvLimit,startParm, indOneMio, prFill,NominalAmt)

RefData                 = struct();
RefData.Predictors      = single(Predictors);
RefData.Price           = (Price);
RefData.cost            = single(max(Spread,1*pipette)/2+Brokerage);
RefData.Brokerage       = Brokerage;
RefData.onePip          = onePip;
RefData.Time            = Time;
RefData.InvLimit        = InvLimit;
RefData.indOneMio       = indOneMio;
RefData.prFill          = prFill;



RefData.medianPrice     = median(Price);

% RefData.Spread = single(Spread);
% RefData.ESpread = mean(Spread);


display('Starting GA');

format long


display('save data to disk')


RefID = savesharedata(RefData,'myGAData');
%% to read the data it will create global noteletCache: remember to clear it at the end of the file


display('initialize matlabpool');

	if matlabpool('size') > 0
	      matlabpool close;
    end

    
    workers = 8;
    matlabpool('local', workers);


%------------------------        parameters        ------------------------
% befor using this function you must specified your function in fun00.m
% file in current directory and then set the parameters

var = 4;


n=60 ;            % Number of population

m0=20;            % Number of generations that max value remains constant
                  %   (use for termination criteria)

                  
nMaxGen   =200;%280;                %maximum number of generations/default 1e4                  
nmutationG=1+floor(.05*n);     %number of mutation children(Gaussian)
nmutationR=1+floor(.15*n);     %number of mutation children(random)

nelit=1+floor(.05*n);           %number of elitism children: about 5% population

% % valuemin=[1 1 .25 0];     % min possible value of variables
% % valuemax=[20 20 1.25 1.25];      % max possible value of variables
% % 

% % valuemin=[1 1 .25 0];     % min possible value of variables
% % valuemax=[20 20 2.50 1.25];      % max possible value of variables


valuemin=[1 1 5*1e-2 1];           % min possible value of variables
valuemax=[20 20 3 1.5];      % max possible value of variables

valuemin=[1 1  1e-6 .5 ];           % min possible value of variables
valuemax=[20 20 10*RefData.medianPrice*onePip  1.5 ];      % max possible value of variables


% valuemin=[1 1  1e-6*RefData.medianPrice .5 ];           % min possible value of variables
% valuemax=[20 20 1*RefData.medianPrice  1.5 ];      % max possible value of variables



% valuemin=[1 1, .0001 .75, 1 1  ];           % min possible value of variables
% valuemax=[20 20, 5   1.5, 20 20  ];      % max possible value of variables





valueminMat= repmat(valuemin,n,1);
valuemaxMat= repmat(valuemax,n,1);



%-------------------------------------------------------------------------
nmutation=  nmutationG+nmutationR;
sigma    = (valuemax-valuemin)/5;    %Parameter that related to Gaussian
sigmaMat = (valuemaxMat-valueminMat)/5;

%   function and used in mutation step
max1=zeros(nelit,var);
parent=zeros(n,var);
%cu=[valuemin(1) valuemax(1) valuemin(2) valuemax(2)];


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

                    
                   
                    
%-------------   ****    termination criteria   ****-------------
while m<nMaxGen 
    
%     abs(maxvalue(m)-maxvalue(m-(m0-1)))>0.001*maxvalue(m) &...
%         (abs(maxvalue(m))>1e-10 & abs(maxvalue(m-(m0-1)))>1e-10)...
%         & m<nMaxGen & abs(maxvalue(m)-meanvalue(m))>1e-5 | m<20

    
    ii=ii+1;
    display(ii)
    
    
%------------------- reducing the sigma value
    
     sigmaDecay_gs = sigmaDecay^(1+gs);

     sigma         = sigma./sigmaDecay_gs;% reducing the sigma value
     sigmaMat      = sigmaMat/sigmaDecay_gs;

     
     
     %  ------  **** % reducing the number of mutation()random   **** ----
%     g=g+1;
%     
%     if g>10 && nmutationR>1
%         g=0;
%         nmutationR=nmutationR-1;
%         nmutation=nmutationG+nmutationR;
%     end


    gs=gs+1;
    
    if gs>10 %&& nmutationR>0
        gs=0;
    end

    
    %-------------   ****    function evaluation   ****-------------
    tic  
  parfor i=1:n
      
%        y(i)=fun00OLSparAS5(p(i,:), RefID);
        y(i)=fun00OLSparAS6(p(i,:), RefID);
 
%        y(i)=fun00OLSparAS2delay(p(i,:), RefID);
 
  
  
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
  
   
   
     IndBest  = find(y == maxvalue1(1));
     pBest    = p(IndBest(1),:);
   
   
   
   
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
        
 p((nXover+1):(n-nmutation),:) = max1; 
 
 
 %-------------   ****    mutation   ****------------  
  
    Ind  = [(n-nmutation+1):1:(n-nmutationR)];
    nInd = length(Ind);    
        
    phi         =  1-2*rand(nInd,var);
    z           =  erfinv(phi)*(2^0.5);
    p(Ind,:)    =  z.*sigmaMat(Ind,:)+parent(Ind,:);
               
    Ind      = [n-nmutationR+1:1:n];
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
  p =  single(p);
end

matlabpool close;

clear global % noteletCache ; % these are the data created using the readsharedata in parfor function

%clc
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
