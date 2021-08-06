#define _XOPEN_SOURCE
#define _DEFAULT_SOURCE
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <stdlib.h>
#include <stdio.h>
#include <ia32intrin.h>
#include <cmath>

#include "euro_opt.h"

std::vector<std::string > parseCSV(std::string file_name)
{
    std::string file = "../../../data/gpairs/" + file_name;
    std::ifstream  data(file);
    std::string line;
    std::vector<std::string> parsedCsv;
    while(std::getline(data,line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        while(std::getline(lineStream,cell,','))
        {
            parsedCsv.push_back(cell);
        }

    }
    return parsedCsv;
}

void load_array(int npoints, tfloat **array, std::string file_name) {
    std::vector<std::string> parsed_csv = parseCSV(file_name);

    for (int i = 0; i < npoints; i++) {
        (*array)[i] = std::stof(parsed_csv[i]);
    }
}

void InitData(size_t npoints, tfloat **x1, tfloat **y1, tfloat **z1, tfloat **w1,
	      tfloat **x2, tfloat **y2, tfloat **z2, tfloat **w2, tfloat **rbins, tfloat **results_test) {

    /* Allocate aligned memory */
    *x1 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
    *y1 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
    *z1 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
    *w1 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
    *x2 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
    *y2 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
    *z2 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
    *w2 = (tfloat*)_mm_malloc(npoints * sizeof(tfloat), ALIGN_FACTOR);
    *rbins = (tfloat*)_mm_malloc(DEFAULT_NBINS * sizeof(tfloat), ALIGN_FACTOR);
    *results_test = (tfloat*)_mm_malloc((DEFAULT_NBINS-1) * sizeof(tfloat), ALIGN_FACTOR);

    if ( (*x1 == NULL) || (*y1 == NULL) || (*z1 == NULL) || (*w1 == NULL) ||
       (*x2 == NULL) || (*y2 == NULL) || (*z2 == NULL) || (*w2 == NULL)) {
        printf("Memory allocation failure\n");
        exit(-1);
    }

    load_array(npoints, x1, "x1.csv");
    load_array(npoints, y1, "y1.csv");
    load_array(npoints, z1, "z1.csv");
    load_array(npoints, w1, "w1.csv");
    load_array(npoints, x2, "x2.csv");
    load_array(npoints, y2, "y2.csv");
    load_array(npoints, z2, "z2.csv");
    load_array(npoints, w2, "w2.csv");

    load_array(DEFAULT_NBINS, rbins, "rbins_squared.csv");

    for (unsigned int i = 0; i < DEFAULT_NBINS-1; i++) {
	(*results_test)[i] = 0;
    }
}

/* Deallocate arrays */
void FreeData( tfloat *x1, tfloat *y1, tfloat *z1, tfloat *w1,
	       tfloat *x2, tfloat *y2, tfloat *z2, tfloat *w2, tfloat *rbins, tfloat *results_test )
{
    /* Free memory */
    _mm_free(x1);
    _mm_free(y1);
    _mm_free(z1);
    _mm_free(w1);
    _mm_free(x2);
    _mm_free(y2);
    _mm_free(z2);
    _mm_free(w2);
    _mm_free(rbins);
    _mm_free(results_test);
}

/*
int main()
{
    int npoints = 10;
    tfloat *x1, *y1, *z1, *w1, *x2, *y2, *z2, *w2, *rbins, *results_test;
    InitData(npoints, &x1, &y1, &z1, &w1, &x2, &y2, &z2, &w2, &rbins, &results_test);

    for (unsigned int i = 0; i < npoints; i++) {
        std::cout << x1[i] << std::endl;
    }

    return 0;
}
*/
