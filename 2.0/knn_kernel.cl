/*
 * KNN kernel file ver 2.0
 */

/*
 * distance measure
 */
float calculate_distance(const __global float * test, const __global float * train, int test_sample, int train_sample){
    float res = 0;
    float tmp;

    #pragma unroll D
    for (int i = 0; i < D; i++) {
        tmp = test[i * Q + test_sample] - train[i * N + train_sample];
        res += tmp * tmp;
    }
    return res;
}

/*
 * procedure finds k smallest elements of 
 * the two parts of given table
 */
void merge(int startA, int startB, __local float * neigh, 
          __local unsigned int * labs, int group_size)
{
    #pragma unroll K
    for (int ii = 0; ii < K; ++ii) {
        int i = startB + ii;
        float maxDist = -1;
        int idx = 0;

        #pragma unroll K
        for (int jj = 0; jj < K; ++jj) {
            int j = startA + jj;
            if (neigh[j] > maxDist) {
                idx = j;
                maxDist = neigh[j];
            }
        }
        
        if (maxDist > neigh[i]) {
            neigh[idx] = neigh[i];
            labs[idx] = labs[i];
        }
    }
}    

__kernel void templateKernel(const __global float* train_data,
                             const __global unsigned int * labels,
                             const __global float* test_data,
                             __local float* neighbours,
                             __local unsigned int * labs,
                             __global unsigned int * decisions
                             )
{
    float tmp = 0;
    float tmp_dist = 0;
    int tmp_idx = 0;

    float furthest_dist = 0;
    int furthest_idx_neigh = 0;
    
    uint group_tid = get_local_id(0);
    uint group_size = get_local_size(0);
    uint groupID = get_group_id(0); 
   
    #pragma unroll K
    for(int i = 0; i < K; i++){
        neighbours[group_tid * K + i] = INT_MAX;
    }
    
    if(group_tid < N){
/* 
 * each thread from work-group is working with training object
 * that has id = group_tid + group_size * i, where id < n; i \in N
 */
        for(int train_row = group_tid; train_row < N; 
                train_row += group_size){
        
            //index in private neighbours distances
            int idx = (train_row - group_tid)/group_size;

            tmp_dist = calculate_distance(test_data, train_data, 
                groupID, train_row);
        
            // for the first k samples, remember them as nearest 
            if(idx < K){    
                int tmp_idx = group_tid * K + idx;
            
                // and control which is the furthest one
                if(tmp_dist > furthest_dist){
                    furthest_dist = tmp_dist;
                    furthest_idx_neigh = tmp_idx;
                }

                neighbours[tmp_idx] = tmp_dist;
                labs[tmp_idx] = labels[train_row];

            // if "distance from object < furthest" save it
            } else {

                if(tmp_dist < furthest_dist){
                    neighbours[furthest_idx_neigh] = tmp_dist;
                    labs[furthest_idx_neigh] = labels[train_row];

                    // find new furthest neighbour
                    furthest_dist = neighbours[group_tid * K];
                    furthest_idx_neigh = group_tid * K;
                
                    #pragma unroll K
                    for(int i = 0; i < K; ++i){
                        int tmp_idx = group_tid * K + i;
                        if(furthest_dist < neighbours[tmp_idx]){
                            furthest_dist = neighbours[tmp_idx];
                            furthest_idx_neigh = tmp_idx;
                        }   
                    }
                }
            }
        }

/*
 * we need to wait for each thread in group to end his
 * computation
 */
        barrier (CLK_LOCAL_MEM_FENCE);

 /*
 * here we know that distances for each thread are computed,
 * so we can start merging them to find k nearest neighbours -
 * - globally among all train samples.
 * we do it in 'tournament-way'.
 * The final group is stored in the first k cells of neighbours table
 * and their corresponding labels are in the labs table.
 */ 
        for(int i = 2; i <= group_size; i <<= 1){
            if(group_tid % i == 0 && 
                    (group_tid + (i >> 1) ) < group_size ){
                merge(group_tid * K, (group_tid + (i >> 1 ) ) * K, 
                    neighbours, labs, group_size);
            }
            barrier (CLK_LOCAL_MEM_FENCE);
        }

/* 
 * voting is managed by just one thread
 */
        if(group_tid == 0){

            // we can use neighbours table as the place for voting
            for(int i = 0 ; i < L; i++){
                neighbours[i] = 0;
            }

            // each neighbour gives his vote for one label
            #pragma unroll K
            for(int i = 0 ; i < K; i++){
                neighbours[labs[i]]+=1;
            }

            int max_votes = 0;
            int decision = 0;

            // find the label having the most voices
            // (when ties choose the lowest one as expected)
            for(int i = 0; i < L; i++){
                if(max_votes < neighbours[i]){
                    max_votes = neighbours[i];
                    decision = i;
                }
            }     
    
            // save decision
            decisions[groupID] = decision;

        }

    }
} 
