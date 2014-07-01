#include <string.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <cstdlib>
#include <limits.h>

using namespace std;

/*
 * a bunch of print functions
 */
void printArray(float ** tab, int h, int w){
    for(int i = 0; i < h; i++){
        for(int j =0; j < w; j++){
            printf("%f, ", tab[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printTab(int * tab, int size){
    for (int i = 0 ; i < size; i++){
        printf("%d, ", tab[i]);
    }
    printf("\n");
}

void printTabF(float * tab, int size){
    for (int i = 0 ; i < size; i++){
        printf("%f, ", tab[i]);
    }
    printf("\n");
}

/*
 * distance measure
 */
float dist(float * x, float * y, int d){
    float res = 0;
    for(int i = 0; i < d; i++){
       res += (x[i] - y[i]) * (x[i] - y[i]) ;
    }
    return res;
}

/*
 * comparator
 */
int compare (const void * a, const void * b){
  return ( *(int*)a - *(int*)b );
}

/*
 * save the decisions which KNN has given to the output file
 */
void saveDecisions(int * decisions, int size, char * output){
    FILE * ofp = fopen(output, "w");

    if (ofp == NULL) {
        fprintf(stderr, "Can't open output file %s!\n", output);
        exit(1);
    }

    for(int i = 0; i < size; i++){
        fprintf(ofp, "%d\n", decisions[i]);
    }

    fclose(ofp);
}

int main(int argc, char * argv[]){
    int sizeInBytes;
    int n, d, l, q, k;
    int * labels, * votes, * decisions;
    float ** train_data,** test_data;
    float ** distance;
    FILE * ifd = fopen(argv[1], "r");

    if (ifd == NULL) {
        fprintf(stderr, "Can't open input file in.list!\n");
        exit(1);
    }

 
    fscanf(ifd, "%d %d %d %d %d\n", &n, &d, &l, &q, &k);

    sizeInBytes = q * sizeof(float*);
    distance = (float **) malloc(sizeInBytes);
    sizeInBytes = d * sizeof(float);
    for(int i = 0; i < n; i++){
        distance[i] = (float *) malloc(sizeInBytes);
    }

    if(distance == NULL)
	{
		std::cout<<"Error: Failed to allocate distance memory on host\n";
		return 1; 
	}

    sizeInBytes = l * sizeof(int);
    votes = (int *) malloc(sizeInBytes);
    if(votes == NULL)
	{
		std::cout<<"Error: Failed to allocate votes memory on host\n";
		return 1; 
	}
    
    sizeInBytes = q * sizeof(int);
    decisions = (int *) malloc(sizeInBytes);
    if(decisions == NULL)
	{
		std::cout<<"Error: Failed to allocate decisions memory on host\n";
		return 1; 
	}

    sizeInBytes = n * sizeof(int);
    labels = (int *) malloc(sizeInBytes);
    if(labels == NULL)
	{
		std::cout<<"Error: Failed to allocate labels memory on host\n";
		return 1; 
	}

    sizeInBytes = n * sizeof(float*);
    train_data = (float **) malloc(sizeInBytes);
    sizeInBytes = d * sizeof(float);
    for(int i = 0; i < n; i++){
        train_data[i] = (float *) malloc(sizeInBytes);
    }
    if(train_data == NULL)
	{
		std::cout<<"Error: Failed to allocate train_data memory on host\n";
		return 1; 
	}

    sizeInBytes = q * sizeof(float*);
    test_data = (float **) malloc(sizeInBytes);
	sizeInBytes = d * sizeof(float);
    for(int i = 0; i < q; i++){
        test_data[i] = (float *) malloc(sizeInBytes);
    }    
    if(test_data == NULL)
	{
		std::cout<<"Error: Failed to allocate test_data memory on host\n";
		return 1; 
	}    

    for(int i = 0; i < n; i++){
        fscanf(ifd, "%d ", &labels[i]);
        for(int j = 0; j < d; j++){
            fscanf(ifd, "%f ", &train_data[i][j]);
        }
    }
    
    for(int i = 0; i < q; i++){
        for(int j = 0; j < d; j++){
            fscanf(ifd, "%f ", &test_data[i][j]);
        }
    }

    // wyznacz macierz odleglosci
    for(int i = 0; i < q; i++){
        for(int j = 0 ; j < n; j++){
            distance[i][j] = dist(test_data[i], train_data[j], d);
        }
    }

    const float USED = -1;
    // dla kazdego obiektu testowego
    for(int i = 0; i < q; i++){
        float min = INT_MAX;
        int max = INT_MIN;
        int max_idx = -1;

        // wybierz k obiektow treningowych
        for(int j = 0; j < k; j++){
            int min_idx = -1;
            min = INT_MAX;
            
            // najblizszych wg odleglosci
            for(int m = 0; m < n; m++){
                if (min > distance[i][m] && distance[i][m] != USED) {
                    min = distance[i][m];
                    min_idx = m;
                }
            }

            // j-ty najblizszy glosuje
            votes[labels[min_idx]] += 1;
            distance[i][min_idx] = USED;
        }

        // podsumuj wyniki glosowania, zadecyduj
        for(int j = 0; j < l; j++){
            if(max < votes[j]){
                max = votes[j];
                max_idx = j;
            }
            votes[j] = 0;
        }
        
        decisions[i] = max_idx;
    }

    saveDecisions(decisions, q, argv[2]);
}


