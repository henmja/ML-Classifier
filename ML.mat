%Class 1 = w1, class 2 = w2

%Reclassification using w1 (120 columns): 
load('project.mat'); 
my=mean(Dtrain{1}')'; 
my2=mean(Dtrain{2}')'; 
test = Dtrain{1}; 
Sgm = cov(Dtrain{1}'); 
Sgm2 = cov(Dtrain{2}'); 
[PC,g,c]=clf(my,my2,Sgm,Sgm2,test) 
%Reclassification using w2 (120 columns):
test = Dtrain{2}; 
clf(my,my2,Sgm,Sgm2,test) 
%Classification using w1 (80 columns):
 test = Dtest{1} 
clf(my,my2,Sgm,Sgm2,test) 

%Classification using w2 (80 columns):
test = Dtest{2} 
clf(my,my2,Sgm,Sgm2,test) 


function [PC,g,c]=clf(my,my2,Sgm,Sgm2,test)
x1=test(1,:)
x2=test(2,:)

d = length(my);
d2 = length(my2);
[X1,Y1] = meshgrid(x1,x2);
% Preallocate the memory:
sX=size(x1,1);
sY=size(x2,2);
p = zeros(sX,sY);
for i=1:sX
    for j=1:sY
        x=test(:,j);
        p(i,j)=1/((2*pi)^(d/2)*det(Sgm)^(1/2))*exp(-1/2*((x-my)'*inv(Sgm)*(x-my)));
    end
end

% Preallocate the memory:
p2 = zeros(sX,sY);
for i=1:sX
    for j=1:sY
        x=test(:,j);
        p2(i,j)=1/((2*pi)^(d2/2)*det(Sgm2)^(1/2))*exp(-1/2*((x-my2)'*inv(Sgm2)*(x-my2)));
    end
end

maximum = max([p;p2]) 
correct = 0;
for i=1:length(p)
    if p(:,i)==maximum(1,i)
        c(1,i)=1;
        p(:,i)
        p2(:,i)
        if (sY==120) 
            correct = correct+1
        end
    end
    if p2(:,i)==maximum(1,i)
        c(1,i)=2;
        if (sY==80) 
            correct = correct+1
        end
    end
end

g(1,:)=p;
g(2,:)=p2;

PC = correct*100/sY %P|correct
ER = 100-PC %P|Error
end
