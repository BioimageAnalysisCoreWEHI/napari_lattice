// Coverslip-frame orthogonal deskew — X-skew (single-pass inverse gather).
//
// Mirrors the Y-skew coverslip kernel with the sheared axis = X.
// The raw image has shape (Nz=scan, Ny, Nx); the shear direction is X.
//
// FROZEN MAP (mirrors numba X-skew: axis swap 1<->2 then Y-kernel then swap):
//   Inverse gather (per output voxel xc, yc, zc):
//     ss    = sintheta * pixel_step  (pixel_step = dz/dx)
//     plane = zc / ss               (scan-plane index, float)
//     raw_x = xc - zc / tantheta   (raw-X column index, float)
//     y     = yc                    (pass-through)
//   Bilinear over nearest (plane, raw_x) neighbours; yc passes through.
//   No zc flip; z0 = x0 = 0.
//
// OpenCL image layout for a numpy (Nz=scan, Ny, Nx) array pushed via cle:
//   READ  raw[plane, yc, raw_x]  -> (int4)(raw_x, yc, plane, 0)
//   WRITE out[zc, yc, xc]        -> (int4)(xc, yc, zc, 0)
//
// THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.

#ifndef SAMPLER_ADDRESS
#define SAMPLER_ADDRESS CLK_ADDRESS_CLAMP
#endif

__kernel void
affine_transform_deskew_x_shear_only_3d(IMAGE_input_TYPE  input,
                                       IMAGE_output_TYPE output,
                                       float pixel_step,
                                       float tantheta,
                                       float sintheta)
{
    const sampler_t sampler =
        CLK_NORMALIZED_COORDS_FALSE | SAMPLER_ADDRESS | CLK_FILTER_NEAREST;

    // Output voxel indices (coverslip frame)
    const uint xc = get_global_id(0);   // sheared X (coverslip X)
    const uint yc = get_global_id(1);   // Y passthrough
    const uint zc = get_global_id(2);   // height above coverslip (coverslip Z)

    // Bounds of the RAW (input) image
    const uint Nz = GET_IMAGE_DEPTH(input);   // number of scan planes
    const uint Nx = GET_IMAGE_WIDTH(input);   // raw-X columns

    float pix = 0.0f;

    if (xc < GET_IMAGE_WIDTH(output) &&
        yc < GET_IMAGE_HEIGHT(output) &&
        zc < GET_IMAGE_DEPTH(output))
    {
        // Inverse map: find the fractional scan-plane and raw-X indices
        const float ss    = sintheta * pixel_step;   // zc spacing per scan plane
        const float scan  = (float)zc / ss;          // fractional scan-plane index
        const long  plane = (long)floor(scan);       // floor scan plane
        const float fp    = scan - (float)plane;     // fraction between planes

        const float raw_x = (float)xc - (float)zc / tantheta;
        const long  pos   = (long)floor(raw_x);      // floor raw-X column
        const float fx    = raw_x - (float)pos;      // fraction between columns

        // Only fill voxels whose source lies within the raw volume
        if (plane >= 0 && plane + 1 < (long)Nz &&
            pos   >= 0 && pos   + 1 < (long)Nx)
        {
            // Bilinear interpolation over (plane, plane+1) x (pos, pos+1)
            // raw array layout (Nz=scan, Ny, Nx):
            //   READ raw[p, yc, r]  ->  (int4)(r, yc, p, 0)
            const float p00 = (float)(READ_input_IMAGE(
                input, sampler, (int4)(pos,     yc, plane,     0)).x);
            const float p01 = (float)(READ_input_IMAGE(
                input, sampler, (int4)(pos + 1, yc, plane,     0)).x);
            const float p10 = (float)(READ_input_IMAGE(
                input, sampler, (int4)(pos,     yc, plane + 1, 0)).x);
            const float p11 = (float)(READ_input_IMAGE(
                input, sampler, (int4)(pos + 1, yc, plane + 1, 0)).x);

            pix = (1.0f - fp) * (1.0f - fx) * p00
                + (1.0f - fp) *        fx    * p01
                +        fp   * (1.0f - fx)  * p10
                +        fp   *        fx    * p11;
        }
    }

    // Write to output: WRITE out[zc, yc, xc] -> (int4)(xc, yc, zc, 0)
    WRITE_output_IMAGE(output, (int4)(xc, yc, zc, 0),
                       CONVERT_output_PIXEL_TYPE(pix));
}
