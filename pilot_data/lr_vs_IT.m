clear all
lrs =[0.1,0.2,0.25,0.3]; %lrates
values=[-1,1,1,1,-1]; %values
value=0; %starting value
inv_temp=[1,2,5,10]; %inverse temperatures
trials_learning=[1,3,6,9,12,15,18,21,24];
associability=[];
trials_testing=20;
counter=0;
for lr=lrs % consider a possible learning rate
    for counter=1:9 %consider a number of learning trials
        for c=1:100
            value=0;
            for j=1:trials_learning(counter)
                if rand<0.6;
                    obs=1;
                else
                    obs=0;
                end
                value=value+(lr*(obs-value));
            end
            vc(c)=value;
        end
        value=mean(vc); %average estimate of the transition probability for given # trials and LR.

        it_counter=1;
        for temp=inv_temp %consider a given inverse temperature
            temp;
            
            for k=1:100
                bc=0; %best-choice counter
                for i=1:trials_testing
                    pc=1/(1+exp(-temp*value));
                    if rand<pc
                        choice=1;
                        bc=bc+1;    
                    end
                end
                bcs(k)=bc;
            end
            bc=mean(bcs);
            if lr==0.1
                lr_lowest(it_counter,counter)=bc;
            elseif lr==0.2
                lr_low(it_counter,counter)=bc;
            elseif lr==0.25
                lr_mid(it_counter,counter)=bc;
            else
                lr_high(it_counter,counter)=bc;
            end
            it_counter=it_counter+1;
        end
    end    
end

X=[1,3,6,9,12,15,18,21,24];

lr_mat(1,1:9)=lr_lowest(1,1:9);
lr_mat(2,1:9)=lr_low(1,1:9);
lr_mat(3,1:9)=lr_mid(1,1:9);
lr_mat(4,1:9)=lr_high(1,1:9);

lr_mat2(1,1:9)=lr_lowest(3,1:9);
lr_mat2(2,1:9)=lr_low(3,1:9);
lr_mat2(3,1:9)=lr_mid(3,1:9);
lr_mat2(4,1:9)=lr_high(3,1:9);

figure
subplot(4,1,1);
plot(X,lr_lowest(1,1:9),X,lr_lowest(2,1:9),X,lr_lowest(3,1:9),X,lr_lowest(4,1:9));
ylim([6,21]);
legend('temp=1','temp=2.0','temp=3.0','temp=4.0');
title('Varying Inv Temp while LR=0.1')
subplot(4,1,2);
plot(X,lr_mid(1,1:9),X,lr_mid(2,1:9),X,lr_mid(3,1:9),X,lr_mid(4,1:9));
ylim([6,21]);
legend('temp=1','temp=2.0','temp=3.0','temp=4.0');
title('Varying Inv Temp while LR=0.25')
subplot(4,1,3);
plot(X,lr_mat(1,1:9),X,lr_mat(2,1:9),X,lr_mat(3,1:9),X,lr_mat(4,1:9));
ylim([6,21]);
legend('lr=0.1','lr=0.2','lr=0.25','lr=0.3');
title('Varying LR while Inv Temp=1')
subplot(4,1,4);
plot(X,lr_mat2(1,1:9),X,lr_mat2(2,1:9),X,lr_mat2(3,1:9),X,lr_mat2(4,1:9));
ylim([6,21]);
legend('lr=0.1','lr=0.2','lr=0.25','lr=0.3');
title('Varying LR while Inv Temp=5')
        