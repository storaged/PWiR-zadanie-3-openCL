/*
 * KNN kernel file ver 1.0
 */

float calculate_distance(const __global float * x, const __global float * y, int tid, int row, int d){
    float res = 0;
    float tmp;
    for (int i = 0; i < d; i++) {
        tmp = x[tid * d + i] - y[row * d + i];
        res += tmp * tmp;
    }
    return res;
}


__kernel void templateKernel(const __global float* train_data,
                             const __global unsigned int * labels,
                             const __global float* test_data,
                             __local float* neighbours,
                             __local unsigned int * labs,
                             const unsigned int n,
                             const unsigned int d,
                             const unsigned int l, 
                             const unsigned int q, 
                             const unsigned int k,
                             __local unsigned int * votes,
                             __global unsigned int * decisions
                             )
{
    float tmp = 0;
    float tmp_dist = 0;
    unsigned int tmp_idx = 0;

    float furthest_dist = 0;
    unsigned int furthest_idx_neigh = 0;
    
    uint tid = get_global_id(0);
   
    for(int i = 0 ; i < l; i++){
        votes[i] = 0;
    }

    // for each object in train set
    for(int train_row = 0; train_row < n; train_row++){
        
        tmp_dist = calculate_distance(test_data, train_data, tid, train_row, d);
        
        // for the first k samples, remember them as nearest 
        if(train_row < k){
                
            // and control which is the furthest one
            if(tmp_dist > furthest_dist){
                furthest_dist = tmp_dist;
                furthest_idx_neigh = train_row;
            }

            neighbours[train_row] = tmp_dist;
            labs[train_row] = labels[train_row];

        // if "distance from object < furthest" save it
        } else {
            
            if(tmp_dist < furthest_dist){
                neighbours[furthest_idx_neigh] = tmp_dist;
                labs[furthest_idx_neigh] = labels[train_row];

                // find new furthest neighbour
                furthest_dist = neighbours[0];
                furthest_idx_neigh = 0;
                for(int i = 0; i < k; i++){
                    if(furthest_dist < neighbours[i]){
                        furthest_dist = neighbours[i];
                        furthest_idx_neigh = i;
                    }   
                }
            }
        }
    }

    // do the voting for the label
    for(int i = 0 ; i < k; i++){
        ++votes[labs[i]];
    }

    unsigned int max_votes = 0;
    unsigned int decision = 0;

    // get decisions
    for(int i = 0; i < l; i++){
        if(max_votes < votes[i]){
            max_votes = votes[i];
            decision = i;
        }
    }     

    decisions[tid] = decision;
} 
