function rez = preprocessDataSubH5(ops)

tic;
ops.nt0 = getOr(ops, {'nt0'}, 61);
ops.nt0min = getOr(ops, 'nt0min', ceil(20 * ops.nt0/61));

NT = ops.NT;

NchatTOT = ops.NchanTOT;
finfo = h5info(ops.fbinary, '/sig');
nTimepoints = finfo.Dataspace.Size(1);

ops.tstart  = ceil(ops.trange(1) * ops.fs); % starting timepoint for processing data segment
ops.tend    = min(nTimepoints, ceil(ops.trange(2) * ops.fs)); % ending timepoint
ops.sampsToRead = ops.tend-ops.tstart; % total number of samples to read
ops.twind = ops.tstart;

Nbatch = ceil(ops.sampsToRead/(NT-ops.ntbuff));
ops.Nbatch = Nbatch;

[chanMap, xc, yc, kcoords, NchanTOTdefault] = loadChanMap(ops.chanMap); % function to load channel map file
ops.NchanTOT = getOr(ops, 'NchanTOT', NchanTOTdefault); % if NchanTOT was left empty, then overwrite with the default

if getOr(ops, 'minfr_goodchannels', .1)>0 % discard channels that have very few spikes
    % determine bad channels
    fprintf('Time %3.0fs. Determining good channels.. \n', toc);
    igood = get_good_channelsH5(ops, chanMap);

    chanMap = chanMap(igood); %it's enough to remove bad channels from the channel map, which treats them as if they are dead

    xc = xc(igood); % removes coordinates of bad channels
    yc = yc(igood);
    kcoords = kcoords(igood);

    ops.igood = igood;
else
    ops.igood = true(size(chanMap));
end

ops.Nchan = numel(chanMap); % total number of good channels that we will spike sort
ops.Nfilt = getOr(ops, 'nfilt_factor', 4) * ops.Nchan; % upper bound on the number of templates we can have

rez.ops         = ops; % memorize ops

rez.xc = xc; % for historical reasons, make all these copies of the channel coordinates
rez.yc = yc;
rez.xcoords = xc;
rez.ycoords = yc;
% rez.connected   = connected;
rez.ops.chanMap = chanMap;
rez.ops.kcoords = kcoords;


NTbuff      = NT + 4*ops.ntbuff; % we need buffers on both sides for filtering

rez.ops.Nbatch = Nbatch;
rez.ops.NTbuff = NTbuff;
rez.ops.chanMap = chanMap;


fprintf('Time %3.0fs. Computing whitening matrix.. \n', toc);

% this requires removing bad channels first
Wrot = get_whitening_matrixH5(rez); % outputs a rotation matrix (Nchan by Nchan) which whitens the zero-timelag covariance of the data

fprintf('Time %3.0fs. Loading raw data and applying filters... \n', toc);

fidW        = fopen(ops.fproc,   'w'); % open for writing processed data

for ibatch = 1:Nbatch
    % we'll create a binary file of batches of NT samples, which overlap consecutively on ops.ntbuff samples
    % in addition to that, we'll read another ops.ntbuff samples from before and after, to have as buffers for filtering
    offset = max(1, ops.twind + ((NT - ops.ntbuff) * (ibatch-1) - 2*ops.ntbuff)); % number of samples to start reading at.
    NTs = min(NTbuff, nTimepoints-offset);
    if offset==0
        ioffset = 0; % The very first batch has no pre-buffer, and has to be treated separately
    else
        ioffset = ops.ntbuff;
    end

    buff = h5read(ops.fbinary, '/sig', [offset 1], [NTs NchanTOT]);
    buff = buff';
    if isempty(buff)
        break; % this shouldn't really happen, unless we counted data batches wrong
    end
    nsampcurr = size(buff,2); % how many time samples the current batch has
    if nsampcurr<NTbuff
        buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr); % pad with zeros, if this is the last batch
    end

    datr    = gpufilter(buff, ops, chanMap); % apply filters and median subtraction

    datr    = datr(ioffset + (1:NT),:); % remove timepoints used as buffers

    datr    = datr * Wrot; % whiten the data and scale by 200 for int16 range

    datcpu  = gather(int16(datr)); % convert to int16, and gather on the CPU side
    fwrite(fidW, datcpu, 'int16'); % write this batch to binary file
end
rez.Wrot    = gather(Wrot); % gather the whitening matrix as a CPU variable

fclose(fidW); % close the files

fprintf('Time %3.0fs. Finished preprocessing %d batches. \n', toc, Nbatch);

rez.temp.Nbatch = Nbatch;