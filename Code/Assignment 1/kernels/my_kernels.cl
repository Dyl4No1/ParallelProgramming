// global implementation
kernel void hist_simpleG(global const uchar* A, global int* H) {
	int id = get_global_id(0);
	int bin_index = A[id];

	atomic_inc(&H[bin_index]);
}

//a very simple Local histogram implementation
kernel void hist_simple(global const uchar* A, global int* B, local int* H, int nr_bins) {
    int id = get_global_id(0);
    int lid = get_local_id(0);

    //assumes that H has been initialised to 0
    int bin_index = (A[id] / 256.0f) * nr_bins;//take value as a bin index

    if (lid < nr_bins)
        H[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_inc(&H[bin_index]);

    if (lid < nr_bins)
        atomic_add(&B[lid], H[lid]);

    //atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

kernel void hist_cum(global int* Hist, global int* cumHist) {

	//get global id and size
	int id = get_global_id(0);
	int size = get_global_size(0);

	//loop to cumulatively add rgb values together
	for (int i = id + 1; i < size; i++)
		atomic_add(&cumHist[i], Hist[id]);
}

kernel void norm_cumHistG(global const int* v_cH, global int* v_ncH) {

	int id = get_global_id(0);
	int size = get_global_size(0);

	float top = v_cH[size-1];
	
	v_ncH[id] = (v_cH[id] / top) * 255;
}

kernel void norm_cumHist(global const int* v_cH, global int* v_ncH, local int* b_cH, local int* b_ncH) {

	int id = get_global_id(0);
	int lid = get_local_id(0);

	b_cH[lid] = v_cH[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	float top = v_cH[get_local_size(0) - 1];
	
	v_ncH[id] = (v_cH[lid] / top) * 255;
	barrier(CLK_LOCAL_MEM_FENCE);

	atomic_add(&v_ncH[lid], b_ncH[lid]);
}

kernel void lookupTable(global const uchar* x, global uchar* y, global int* ncH) {

	int id = get_global_id(0);
	int lid = get_local_id(0);
	y[id] = ncH[x[id]];

}  