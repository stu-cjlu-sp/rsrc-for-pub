function[PF]=LMD(x)

% fs=2000;
% t=0:1/fs:1;
% x=15*(1+cos(40*pi*t)).*cos(600*pi*t)+5*(1+cos(40*pi*t)).*cos(200*t*pi);
% x=cos(2*pi*30*t +0.1*sin(2*pi*10*t))+sin(2*80*t);


%规定输入信号为行向量
if size(x,1)~=1
    x = x';
end
c = x;
N = length(x);


PF = [];
A1=[];
Si=[];
while(1) %loop 1
    
    a = 1;
    
    while(1) %loop 2
        h = c;
        
        maxVec = [];%局部极大值
        minVec = [];%局部极小值
        
        % look for max and min point 找到h中的局部极大值和极小值
        for i = 2: N - 1
            if h (i - 1) < h (i) && h (i) > h (i + 1)
                maxVec = [maxVec i];
            end
            if h (i - 1) > h (i) && h (i) < h (i + 1)
                minVec = [minVec i];
            end
        end
        
        % check if it is residual  判断是否满足loop2的终止条件
        if (length (maxVec) + length (minVec)) < 2
            break;
        end
        
        % handle end point

        %left end point
        if h(1)>0
            if(maxVec(1)<minVec(1))
                yleft_max=h(maxVec(1));
                yleft_min=-h(1);
            else
                yleft_max=h(1);
                yleft_min=h(minVec(1));
            end
        else
            if (maxVec(1)<minVec(1))
                yleft_max=h(maxVec(1));
                yleft_min=h(1);
            else
                yleft_max=-h(1);
                yleft_min=h(minVec(1));
            end
        end
        %right end point
        if h(N)>0
            if(maxVec(end)<minVec(end))
                yright_max=h(N);
                yright_min=h(minVec(end));
            else
                yright_max=h(maxVec(end));
                yright_min=-h(N);
            end
        else
            if(maxVec(end)<minVec(end))
                yright_max=-h(N);
                yright_min=h(minVec(end));
            else
                yright_max=h(maxVec(end));
                yright_min=h(N);
            end
        end
        %get envelop of maxVec and minVec using
        %spline interpolate  使用样条插值法得到maxVec和minVec的包络
        maxEnv=spline([1,maxVec,N],[yleft_max h(maxVec) yright_max],1:N);
        minEnv=spline([1,minVec,N],[yleft_min h(minVec) yright_min],1:N);
%         maxEnv=interp1([1 maxVec N],[yleft_max h(maxVec) yright_max],1:N);
%         minEnv=interp1([1 minVec N],[yleft_min h(minVec) yright_min],1:N);
        
        mm = (maxEnv + minEnv)/2;   %局域均值函数
        aa = abs(maxEnv - minEnv)/2;  %包络均值函数
        

        

        si = (h-mm)./aa;  %调频信号
        a = a.*aa;   %包络信号

        bb = norm(aa - ones(1,length(aa)));
        if(bb < 1000)%如果bb<1000，得到纯调频函数
            break;
        end
        
    end %loop2结束
    A1=[A1;a];
    Si=[Si;si];
    pf = a.*si;
    
    PF = [PF; pf];
    
    % check if it is residual  loop2 中止条件
    if (length (maxVec) + length (minVec)) < 20
        break;
    end
    
    c = c-pf;
    
end

line=size(PF,1);
NN = length(PF(1,:));
n = linspace(0,1,NN);

% figure(5);
% 
% subplot(line+1,1,1),plot(n,x*1000),ylabel('X(t)');
% for i = 2:line+1
% subplot(line+1,1,i),plot(n,PF(i-1,:)*1000),ylabel(['PF_',num2str(i),'1(t)']);
% end
end
% subplot(line+1,1,3),plot(n,PF(2,:)*1000),ylabel('PF_2(t)');
% subplot(line+1,1,4),plot(n,PF(3,:)*1000),ylabel('PF_3(t)');



