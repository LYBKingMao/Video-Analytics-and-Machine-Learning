function Objects = NMS(Objects,threshold)
%This NMS is modified from Practical 5 simple NMS.m, but use similar
%method, this NMS use maxi variable generated from SVMTesting which
%represent the distance to classification line, at first this function
%filter the highly overlap sliding window between two neightbours, then in
%order to suppress the non-maximum value, this function keep looping to
%check overlapping area between one sliding window and others, choose
%sliding window with higher distance.
for i=1:size(Objects,1)-1
    j=i+1;
    area=rectint(Objects(i,:),Objects(j,:))
     if area>threshold
        if Objects(i,5)<Objects(j,5)
            Objects(i,:)=Objects(j,:)
        elseif Objects(i,5)>Objects(j,5)
            Objects(j,:)=Objects(i,:)
        end
    end
end
Objects=unique(Objects,'rows');
for k=1:5
for i=1:size(Objects,1)-1
        for j=i+1:size(Objects,1)
            area=rectint(Objects(i,:),Objects(j,:))
         if area>threshold
            if Objects(i,5)<Objects(j,5)
                Objects(i,:)=Objects(j,:)
            elseif Objects(i,5)>Objects(j,5)
                Objects(j,:)=Objects(i,:)
            end
         end
        end
    end
    Objects=unique(Objects,'rows');
end
end
