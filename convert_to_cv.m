function [cv_img, dim, depth, width_step] = convert_to_cv(img)

    % Exchange rows and columns (handles 3D cases as well)
    img2 = permute( img(:,end:-1:1,:), [2 1 3] );

    dim = [size(img2,1), size(img2,2)];

    % Convert double precision to single precision if necessary

    % Determine image depth
    if( ndims(img2) == 3 && size(img2,3) == 3 )
        depth = 3;
    else
        depth = 1;
    end

    % Handle color images
    if(depth == 3 )
        % Switch from RGB to BGR
        img2(:,:,[3 2 1]) = img2;

        % Interleave the colors
        img2 = reshape( permute(img2, [3 1 2]), [size(img2,1)*size(img2,3) size(img2,2)] );
    end

    % Pad the image
    width_step = size(img2,1) + mod( size(img2,1), 4 );
    img3 = double(zeros(width_step, size(img2,2)));
    img3(1:size(img2,1), 1:size(img2,2)) = img2;

    cv_img = img3;

    % Output to openCV
    %cv_display(cv_img, dim, depth, width_step);
end