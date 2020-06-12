function [C] = lmv(X,H,alpha,aa,bb,cc,dd)
% Returns the number of columns in X
X = double(reshape(cell2mat([X{:}]),aa,bb));
H = double(reshape(cell2mat([H{:}]),cc,dd));

num=size(X,1);
[r,~]=size(H);
%k=length(unique(y));

Sbar=[];
options = optimset( 'Algorithm','interior-point-convex','Display','off');

% eye(n):Return a n*n identity matric
A=2*alpha*eye(r)+2*H*H';
        A=(A+A')/2;
        B=X';
parfor ji=1:num
     ff=-2*B(:,ji)'*H';
% 3st,4st=[] is means no linear constraint,ones(r,1),[] is means Lower Bound and UpB
Z(:,ji)=quadprog(A,ff',[],[],ones(1,r),1,zeros(r,1),ones(r,1),[],options);
end
C=Z';
%  f(j)=(norm(X{j}'-H{j}'*S{j},'fro'))^2+alpha*(norm(S{j},'fro'))^2;
% for ji=1:nv
%     f(ji)/sum(f)
%      Sbar=cat(1,Sbar,1/sqrt(nv)*S{ji}/f(ji));
% end

%[U,Sig,V] = mySVD(Sbar',k);
%
%rand('twister',5489)
%labels=litekmeans(U, k, 'MaxIter', 100,'Replicates',10);%kmeans(U, c, 'emptyaction', 'singleton', 'replicates', 100, 'display', 'off');