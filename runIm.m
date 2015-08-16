function E = runIm(imStr,prm)
    EdgeDetection(imStr,prm.removeEpsilon,prm.maxTurn,prm.nmsFact,prm.splitPoints,prm.minContrast);
    E = im2double(imread('res.png'));
    E = E./max(E(:));
end