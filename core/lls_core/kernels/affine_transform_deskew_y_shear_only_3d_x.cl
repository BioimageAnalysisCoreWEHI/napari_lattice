// Coverslip-frame orthogonal deskew — Y-skew (single-pass inverse gather).
//
// Adapted from pyclesperanto_prototype's affine_transform_deskew_y_3d_x.cl
// (Sapoznik et al. 2020; Maioli 2016) for the coverslip coordinate frame.
//
// FROZEN MAP:
//   Forward: zc = sin*step*p,  yc = cos*step*p + yr,  xc = x
//   Inverse gather (per output voxel xc, yc, zc):
//     ss    = sintheta * pixel_step
//     plane = zc / ss              (scan-plane index, float)
//     raw_y = yc - zc / tantheta  (raw-Y row index, float)
//     x     = xc                  (pass-through)
//   Bilinear over nearest (plane, raw_y) neighbours.
//   No zc flip; z0 = y0 = 0.
//
// OpenCL image layout for a numpy (Nz=scan, Ny, Nx) array pushed via cle:
//   READ  raw[plane, raw_y, x]   -> (int4)(x, raw_y, plane, 0)
//   WRITE out[zc, yc, x]         -> (int4)(x, yc, zc, 0)
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice.
// * Redistributions in binary form must reproduce the above copyright notice.
// THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.

#ifndef SAMPLER_ADDRESS
#define SAMPLER_ADDRESS CLK_ADDRESS_CLAMP
#endif

__kernel void
affine_transform_deskew_y_shear_only_3d(IMAGE_input_TYPE  input,
                                       IMAGE_output_TYPE output,
                                       float pixel_step,
                                       float tantheta,
                                       float sintheta)
{
    const sampler_t sampler =
        CLK_NORMALIZED_COORDS_FALSE | SAMPLER_ADDRESS | CLK_FILTER_NEAREST;

    // Output voxel indices (coverslip frame)
    const uint xc = get_global_id(0);   // X passthrough
    const uint yc = get_global_id(1);   // sheared lateral (coverslip Y)
    const uint zc = get_global_id(2);   // height above coverslip (coverslip Z)

    // Bounds of the RAW (input) image
    const uint Nz = GET_IMAGE_DEPTH(input);   // number of scan planes
    const uint Ny = GET_IMAGE_HEIGHT(input);  // raw-Y rows

    float pix = 0.0f;

    if (xc < GET_IMAGE_WIDTH(output) &&
        yc < GET_IMAGE_HEIGHT(output) &&
        zc < GET_IMAGE_DEPTH(output))
    {
        // Inverse map: find the fractional scan-plane and raw-Y indices
        const float ss    = sintheta * pixel_step;   // zc spacing per scan plane
        const float scan  = (float)zc / ss;          // fractional scan-plane index
        const long  plane = (long)floor(scan);       // floor scan plane
        const float fp    = scan - (float)plane;     // fraction between planes

        const float raw_y = (float)yc - (float)zc / tantheta;
        const long  pos   = (long)floor(raw_y);      // floor raw-Y row
        const float fy    = raw_y - (float)pos;      // fraction between rows

        // Only fill voxels whose source lies within the raw volume
        if (plane >= 0 && plane + 1 < (long)Nz &&
            pos   >= 0 && pos   + 1 < (long)Ny)
        {
            // Bilinear interpolation over (plane, plane+1) x (pos, pos+1)
            // raw array layout (Nz=scan, Ny, Nx):
            //   READ raw[p, r, x]  ->  (int4)(x, r, p, 0)
            const float p00 = (float)(READ_input_IMAGE(
                input, sampler, (int4)(xc, pos,     plane,     0)).x);
            const float p01 = (float)(READ_input_IMAGE(
                input, sampler, (int4)(xc, pos + 1, plane,     0)).x);
            const float p10 = (float)(READ_input_IMAGE(
                input, sampler, (int4)(xc, pos,     plane + 1, 0)).x);
            const float p11 = (float)(READ_input_IMAGE(
                input, sampler, (int4)(xc, pos + 1, plane + 1, 0)).x);

            pix = (1.0f - fp) * (1.0f - fy) * p00
                + (1.0f - fp) *        fy    * p01
                +        fp   * (1.0f - fy)  * p10
                +        fp   *        fy    * p11;
        }
    }

    // Write to output: WRITE out[zc, yc, xc] -> (int4)(xc, yc, zc, 0)
    WRITE_output_IMAGE(output, (int4)(xc, yc, zc, 0),
                       CONVERT_output_PIXEL_TYPE(pix));
}
