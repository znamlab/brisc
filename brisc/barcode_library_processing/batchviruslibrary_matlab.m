%% read in sequences and counts of data collapsed for the uncorrected 12nt tagPBC20
function []=batchvirus_matlab(path)

data.counts=dlmread(sprintf('%s_revnum_VBC_counts.txt', path));
data.reads=int8(char(textread(sprintf('%s_revnum_VBC_seq.txt', path), '%s')));
save(sprintf('%s_editd2_dataBC.mat', path),'data','-v7.3');
%% read in bowtie alignment of these barcodes


%% EDIT DISTANCE 1


positions.x=dlmread([sprintf('%s_bowtiealignment_editd1_BC1.txt', path)]);
positions.y=dlmread([sprintf('%s_bowtiealignment_editd1_BC3.txt', path)]);
clustermatrix.C=sparse(positions.x,positions.y,1);

save(sprintf('%s_editd1_clustermatrix.mat', path),'clustermatrix');

load (sprintf('%s_editd1_clustermatrix.mat', path));
graph=[];
[graph.S,graph.G]=graphconncomp(clustermatrix.C,'Directed','false');
save(sprintf('%s_editd1_graph.mat', path),'graph');

load (sprintf('%s_editd1_graph.mat', path));
x=1:graph.S;
[tf,loc]=ismember(x,graph.G,'R2012a');
collapsedreads=data.reads(loc,:);
collapsedcounts=accumarray(graph.G',data.counts);
[collapsed.counts,ix]=sort(collapsedcounts,'descend');
collapsed.reads=collapsedreads(ix,:);
save(sprintf('%s_editd1_collapsed.mat',path),'collapsed');

%% EDIT DISTANCE 2

positions.x=dlmread([sprintf('%s_bowtiealignment_editd2_BC1.txt', path)]);
positions.y=dlmread([sprintf('%s_bowtiealignment_editd2_BC3.txt', path)]);
clustermatrix.C=sparse(positions.x,positions.y,1);

save(sprintf('%s_editd2_clustermatrix.mat', path),'clustermatrix');

load (sprintf('%s_editd2_clustermatrix.mat', path));
graph=[];
[graph.S,graph.G]=graphconncomp(clustermatrix.C,'Directed','false');
save(sprintf('%s_editd2_graph.mat', path),'graph');

load (sprintf('%s_editd2_graph.mat', path));
x=1:graph.S;
[tf,loc]=ismember(x,graph.G,'R2012a');
collapsedreads=data.reads(loc,:);
collapsedcounts=accumarray(graph.G',data.counts);
[collapsed.counts,ix]=sort(collapsedcounts,'descend');
collapsed.reads=collapsedreads(ix,:);
save(sprintf('%s_editd2_collapsed.mat',path),'collapsed');

%% EDIT DISTANCE 3

positions.x=dlmread([sprintf('%s_bowtiealignment_editd3_BC1.txt', path)]);
positions.y=dlmread([sprintf('%s_bowtiealignment_editd3_BC3.txt', path)]);
clustermatrix.C=sparse(positions.x,positions.y,1);

save(sprintf('%s_editd3_clustermatrix.mat', path),'clustermatrix');

load (sprintf('%s_editd3_clustermatrix.mat', path));
graph=[];
[graph.S,graph.G]=graphconncomp(clustermatrix.C,'Directed','false');
save(sprintf('%s_editd3_graph.mat', path),'graph');

load (sprintf('%s_editd3_graph.mat', path));
x=1:graph.S;
[tf,loc]=ismember(x,graph.G,'R2012a');
collapsedreads=data.reads(loc,:);
collapsedcounts=accumarray(graph.G',data.counts);
[collapsed.counts,ix]=sort(collapsedcounts,'descend');
collapsed.reads=collapsedreads(ix,:);
save(sprintf('%s_editd3_collapsed.mat',path),'collapsed');
