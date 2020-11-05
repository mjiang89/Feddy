#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <tgmath.h>
#include <gmp.h>
#include <pthread.h>
#include <string.h>


int E_FIXED_POINT = 32;
int MAX_USER_NUM;
int MAX_DIMENSION;
int NUM_THREAD = 4;
int REPEAT = 50;


typedef struct _thread_data_t{
    mpz_t ****x_nu;
    mpz_t ****y_nu;
    double ***aggregate_gradient;
    double ****user_gradient;
    mpz_t **p_nu;
    mpz_t N;
    mpz_t N2;
    mpz_t prime_base;
    time_t current_time;
    int pid;
} thread_data_t;


void encode(mpz_t x, double y){
    
    mpz_set_d(x, y * pow(2, E_FIXED_POINT));
    
}

void decode(double *x, mpz_t y){
    mpf_set_default_prec(E_FIXED_POINT + 32);
    mpf_t temp;
    mpf_init(temp);
    mpf_set_z(temp, y);
    mpf_div_2exp(temp, temp, E_FIXED_POINT);
    *x = mpf_get_d(temp);
    mpf_clear(temp);
}

void simpleHash(mpz_t x, time_t y, mpz_t prime_base, mpz_t modulus){ // Our hash function does not need be cryptographic. All we need is a deterministic function that makes sure the base remains to be large (i.e., the x in H(t)^{p_nu}=x^{p_nu} has at least one large prime factor). This does not affect the security because the input of the hash is known to everyone.
    // This function turns the UNIX time to a large integer deterministically by computing prime_base^y
    
    mpz_mul_ui(x, prime_base, y);
    mpz_mod(x, x, modulus);
    
    
}



void generateKeys(mpz_t p_nu[], gmp_randstate_t state, mpz_t modulus){
    mpz_t s_nu_mu[MAX_USER_NUM][MAX_USER_NUM];
    for(int i = 0; i < MAX_USER_NUM; i++){
        for(int j = 0; j < MAX_USER_NUM; j++){
            mpz_init(s_nu_mu[i][j]);
            if(i == j){
                mpz_set_ui(s_nu_mu[i][j], 0);
                continue;
            }
            mpz_urandomb(s_nu_mu[i][j], state, 118);
        }
    }
    
    mpz_t p_nu_mu[MAX_USER_NUM][MAX_USER_NUM];
    for(int i = 0; i < MAX_USER_NUM; i++){
        for(int j = 0; j < MAX_USER_NUM; j++){
            mpz_init(p_nu_mu[i][j]);
            mpz_sub(p_nu_mu[i][j], s_nu_mu[i][j], s_nu_mu[j][i]);
        }
    }
    
    for(int i = 0; i < MAX_USER_NUM; i++){
        mpz_init(p_nu[i]);
        mpz_set_ui(p_nu[i], 0);
        for(int j = 0; j < MAX_USER_NUM; j++){
            mpz_clear(s_nu_mu[i][j]);
            mpz_add(p_nu[i], p_nu[i], p_nu_mu[i][j]);
            mpz_clear(p_nu_mu[i][j]);
        }
        //mpz_mod(p_nu[i], p_nu[i], modulus);
    }
}


void init3Darray(mpz_t ****arr){
    *arr = (mpz_t***)malloc(MAX_USER_NUM * sizeof(mpz_t**));
    //printf("%x\n", *arr);
    for(int i = 0; i < MAX_USER_NUM; i++){
        (*arr)[i] = (mpz_t**)malloc(MAX_DIMENSION * sizeof(mpz_t*));
        for(int j = 0; j < MAX_DIMENSION; j++){
            (*arr)[i][j] = (mpz_t*)malloc(MAX_DIMENSION * sizeof(mpz_t));
            for(int k = 0; k < MAX_DIMENSION; k++){
                mpz_init((*arr)[i][j][k]);
            }
            
        }
    }
}

void clear3Darray(mpz_t ****arr){
    for(int i = 0; i < MAX_USER_NUM; i++){
        for(int j = 0; j < MAX_DIMENSION; j++){
            for(int k = 0; k < MAX_DIMENSION; k++){
                mpz_clear((*arr)[i][j][k]);
            }
            free((*arr)[i][j]);
        }
        free((*arr)[i]);
    }
    free(*arr);
}

void *encrypt_thread(void *arg){
    
    thread_data_t *data = (thread_data_t *)arg;

    mpz_t temp;
    mpz_init(temp);
    
    for(int i = 0; i < MAX_USER_NUM; i++){
        for(int j = 0; j < MAX_DIMENSION; j++){
            for(int k = MAX_DIMENSION * (data->pid) / NUM_THREAD; k < MAX_DIMENSION * (data->pid + 1) / NUM_THREAD; k++){
                //printf("%d,%d,%d\n", i, j, k);
                encode((*(data->x_nu))[i][j][k], (*(data->user_gradient))[i][j][k]);
                mpz_mul(temp, data->N, (*(data->x_nu))[i][j][k]);
                mpz_add_ui((*(data->y_nu))[i][j][k], temp, 1);
                simpleHash(temp, data->current_time, data->prime_base, data->N); // temp = H(t)
                mpz_powm(temp, temp, (*(data->p_nu))[i], data->N); // temp = H(t)^{p_nu[i]} mod N^2
                mpz_mul((*(data->y_nu))[i][j][k], (*(data->y_nu))[i][j][k], temp);
                mpz_mod((*(data->y_nu))[i][j][k], (*(data->y_nu))[i][j][k], data->N2); // y_nu[i][j][k] = final ciphertext to be broadcast
            }
            //printf("(enc) i:%d, j:%d\n", i, j);
        }
    }
    mpz_clear(temp);
    
    pthread_exit(NULL);
}


void *aggregate_thread(void *arg){
    thread_data_t *data = (thread_data_t *)arg;

    mpz_t temp;
    mpz_init(temp);
    
    
    for(int j = 0; j< MAX_DIMENSION; j++){
        for(int k = MAX_DIMENSION * (data->pid) / NUM_THREAD; k < MAX_DIMENSION * (data->pid + 1) / NUM_THREAD; k++){
            for(int i = 0; i < MAX_USER_NUM; i++){
                mpz_set_ui(temp, 1);
                mpz_mul(temp, temp, (*(data->y_nu))[i][j][k]);
            }
            mpz_mod(temp, temp, data->N2);
            mpz_sub_ui(temp, temp, 1);
            mpz_tdiv_q(temp, temp, data->N);
            decode(&((*(data->aggregate_gradient))[j][k]), temp);
        }
        //printf("(agg) j:%d\n", j);
    }
    
    
    
    
    mpz_clear(temp);
    
    pthread_exit(NULL);
    
}


int main(int argc, char **argv){
    
    // This program uses a single machine to simulate multi-user secure aggregation
    
/*
    int user_num[4] = {3, 5, 7, MAX_USER_NUM};
    int dimension[5] = {32, 64, 128, 256, MAX_DIMENSION};
 */
    
    
    if(argc != 3) printf("Syntax error: two arguments need to be provided: # of users and dimension.\n");
    MAX_USER_NUM = atoi(argv[1]);
    MAX_DIMENSION = atoi(argv[2]);
    
    
    srand (time ( NULL));
    
    double** aggregate_gradient = (double**)malloc(MAX_DIMENSION * sizeof(double*));
    for(int i = 0; i < MAX_DIMENSION; i++){
        aggregate_gradient[i] = (double*)malloc(MAX_DIMENSION * sizeof(double));
    }
    
    //double user_gradient[MAX_USER_NUM][MAX_DIMENSION][MAX_DIMENSION];
    double*** user_gradient;
    user_gradient = (double***)malloc(MAX_USER_NUM * sizeof(double**));
    for(int i = 0; i < MAX_USER_NUM; i++){
        user_gradient[i] = (double**)malloc(MAX_DIMENSION * sizeof(double*));
        for(int j = 0; j < MAX_DIMENSION; j++){
            user_gradient[i][j] = (double*)malloc(MAX_DIMENSION * sizeof(double));
            for(int k = 0; k < MAX_DIMENSION; k++){
                user_gradient[i][j][k] = (double)rand()/RAND_MAX*2.0-1.0;//float in range -1 to 1
            }
        }
    }
    
    
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state, time(NULL));
    
    
    mpz_t p,q,N,N2,psub1,qsub1,phiN2;
    mpz_inits(p,q,N,N2,psub1,qsub1,phiN2,NULL);
    mpz_urandomb(p, state, 1024);
    mpz_nextprime(p, p);
    mpz_urandomb(q, state, 1024);
    mpz_nextprime(q, q);
    mpz_sub_ui(psub1, p, 1);
    mpz_sub_ui(qsub1, q, 1);
    mpz_mul(N, p, q);
    mpz_mul(N2, N, N);
    mpz_set(phiN2, N);
    mpz_mul(phiN2, phiN2, psub1);
    mpz_mul(phiN2, phiN2, qsub1);
    
    FILE *f;
    char fileName[20];
    strcpy(fileName, "result.txt");
    
    
    
    f = fopen(fileName, "a");
    fprintf(f, "\n# of user: %d, dimension: %d, # of threads:%d, # of repetition: %d\n", MAX_USER_NUM, MAX_DIMENSION, NUM_THREAD, REPEAT);

    
    fflush(f);
    
    struct timespec start, stop;
    
    long double recordedTime[REPEAT];
    long double sum = 0.0;
    long double mean, stddev, min, max, var;
    
    mpz_t *p_nu = (mpz_t*)malloc(MAX_USER_NUM * sizeof(mpz_t));
    
    // single-threaded keygen.
    
    for(int count = 0; count < REPEAT; count++){
        printf("(gen, user#: %d, dim: %d) count: %d\n", MAX_USER_NUM, MAX_DIMENSION, count);
        if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
          perror( "clock gettime" );
          return EXIT_FAILURE;
        }
        generateKeys(p_nu, state, N);
        if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) {
          perror( "clock gettime" );
          return EXIT_FAILURE;
        }
        recordedTime[count] = (stop.tv_sec - start.tv_sec) + (long double)( stop.tv_nsec - start.tv_nsec)
        / ((long double)(1e9) * MAX_USER_NUM);
    }
    
    min = recordedTime[0];
    max = recordedTime[0];
    sum = 0.0;
    for(int count = 0; count < REPEAT; count++){
        sum += recordedTime[count];
        if (min > recordedTime[count]) min = recordedTime[count];
        if (max < recordedTime[count]) max = recordedTime[count];
    }
    mean = sum / REPEAT;
    sum = 0.0;
    for(int count = 0; count < REPEAT; count++){
        sum += (recordedTime[count] - mean) * (recordedTime[count] - mean);
    }
    
    var = sum / REPEAT;
    stddev = sqrtl(var);
    
    fprintf(f, "(KeyGen) mean: %Lf, std: %Lf, max:%Lf, min:%Lf\n",  mean, stddev, max, min);
    
    fflush(f);
    
    
    
    //mpz_t x_nu[MAX_USER_NUM][MAX_DIMENSION][MAX_DIMENSION];
    mpz_t*** x_nu;
    init3Darray(&x_nu);
    
    
    //mpz_t y_nu[MAX_USER_NUM][MAX_DIMENSION][MAX_DIMENSION];
    mpz_t*** y_nu;
    init3Darray(&y_nu);
    
    
    mpz_t prime_base; // this is the prime base for the hash implementation
    mpz_init(prime_base);
    mpz_urandomb(prime_base, state, 1024);
    mpz_nextprime(prime_base, prime_base);
    
    
    
    // the following portion simulates the individual users who generate y_nu=(1+N)^{[x_nu]}H(t)^{p_nu} mod N^2
    time_t current_time = time(NULL);
    mpz_t temp;
    mpz_init(temp);
    
    
    pthread_t threads[NUM_THREAD];
    
    thread_data_t thread_data[NUM_THREAD];
    
    for(int i = 0; i < NUM_THREAD; i++){
        thread_data[i].pid = i;
        thread_data[i].x_nu = &x_nu;
        thread_data[i].aggregate_gradient = &aggregate_gradient;
        thread_data[i].user_gradient = &user_gradient;
        thread_data[i].y_nu = &y_nu;
        thread_data[i].p_nu = &p_nu;
        thread_data[i].current_time = current_time;
        mpz_inits(thread_data[i].N, thread_data[i].N2, thread_data[i].prime_base, NULL);
        mpz_set(thread_data[i].N, N);
        mpz_set(thread_data[i].N2, N2);
        mpz_set(thread_data[i].prime_base, prime_base);
    }
    
    
    
    
    // Simulation of multi-threaded encryption
    
    
    
    for(int count = 0; count < REPEAT; count++){
        printf("(enc, user#: %d, dim: %d) count: %d\n", MAX_USER_NUM, MAX_DIMENSION, count);
        if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
          perror( "clock gettime" );
          return EXIT_FAILURE;
        }
        for(int i = 0; i < NUM_THREAD; i++){
            int rc;
            if((rc = pthread_create(&(threads[i]), NULL, encrypt_thread, &thread_data[i]))){
                printf("error: pthread_create, rc: %d\n", rc);
                return EXIT_FAILURE;
            }
        }
        
        for(int i = 0; i < NUM_THREAD; i++){
            pthread_join(threads[i], NULL);
        }
        if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) {
          perror( "clock gettime" );
          return EXIT_FAILURE;
        }
        recordedTime[count] = (stop.tv_sec - start.tv_sec) + (long double)( stop.tv_nsec - start.tv_nsec)
        / ((long double)(1e9) * MAX_USER_NUM);
    }
    
    
    
    min = recordedTime[0];
    max = recordedTime[0];
    sum = 0.0;
    for(int count = 0; count < REPEAT; count++){
        sum += recordedTime[count];
        if (min > recordedTime[count]) min = recordedTime[count];
        if (max < recordedTime[count]) max = recordedTime[count];
    }
    mean = sum / REPEAT;
    sum = 0.0;
    for(int count = 0; count < REPEAT; count++){
        sum += (recordedTime[count] - mean) * (recordedTime[count] - mean);
    }
    var = sum / REPEAT;
    stddev = sqrtl(var);
    
    fprintf(f, "(Encryption) mean: %Lf, std: %Lf, max:%Lf, min:%Lf\n", mean, stddev, max, min);
    

    
    fflush(f);
/*
    fclose(f);
    
    return EXIT_SUCCESS;
  */
    
    

    
    
    // Simulation of multi-threaded aggregation
    
    for(int count = 0; count < REPEAT; count++){
        printf("(agg, user#: %d, dim: %d) count: %d\n", MAX_USER_NUM, MAX_DIMENSION, count);
        if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
          perror( "clock gettime" );
          return EXIT_FAILURE;
        }
        for(int i = 0; i < NUM_THREAD; i++){
            int rc;
            if((rc = pthread_create(&(threads[i]), NULL, aggregate_thread, &thread_data[i]))){
                printf("error: pthread_create, rc: %d\n", rc);
                return EXIT_FAILURE;
            }
        }
        for(int i = 0; i < NUM_THREAD; i++){
            pthread_join(threads[i], NULL);
        }
        if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) {
          perror( "clock gettime" );
          return EXIT_FAILURE;
        }
        recordedTime[count] =(stop.tv_sec - start.tv_sec) + (long double)( stop.tv_nsec - start.tv_nsec)
        / (long double)(1e9);
    }
    
    
    min = recordedTime[0];
    max = recordedTime[0];
    sum = 0.0;
    for(int count = 0; count < REPEAT; count++){
        sum += recordedTime[count];
        if (min > recordedTime[count]) min = recordedTime[count];
        if (max < recordedTime[count]) max = recordedTime[count];
    }
    mean = sum / REPEAT;
    sum = 0.0;
    for(int count = 0; count < REPEAT; count++){
        sum += (recordedTime[count] - mean) * (recordedTime[count] - mean);
    }
    var = sum / REPEAT;
    stddev = sqrtl(var);
    
    fprintf(f, "(Aggregation) mean: %Lf, std: %Lf, max:%Lf, min:%Lf\n",  mean, stddev, max, min);
    
    fflush(f);
    

	fclose(f);
	

	

	gmp_randclear(state);
 
    mpz_clears(p,q,N,N2,psub1,qsub1,phiN2,NULL);
    mpz_clear(prime_base);
    
    
    
    for(int i = 0; i < NUM_THREAD; i++){
        mpz_clears(thread_data[i].N, thread_data[i].N2, thread_data[i].prime_base,NULL);
    }
    
    
    for(int i = 0; i < MAX_DIMENSION; i++){
        free(aggregate_gradient[i]);
    }
    free(aggregate_gradient);
    
    for(int i = 0; i < MAX_USER_NUM; i++){
        for(int j = 0; j < MAX_DIMENSION; j++){
            free(user_gradient[i][j]);
        }
        free(user_gradient[i]);
    }
    free(user_gradient);
    
    free(p_nu);
    
    
    clear3Darray(&x_nu);
    clear3Darray(&y_nu);

    
    return EXIT_SUCCESS;
 
}


