#include "mex.h"
#include <math.h>
#include <string.h>
#include <stdint.h>

// 预计算阶乘缓存（使用uint64_t，最大到20!）
static const uint64_t factorial_cache[] = {
    1,                  // 0!
    1,                  // 1!
    2,                  // 2!
    6,                  // 3!
    24,                 // 4!
    120,                // 5!
    720,                // 6!
    5040,               // 7!
    40320,              // 8!
    362880,             // 9!
    3628800,            // 10!
    39916800,           // 11!
    479001600,          // 12!
    6227020800ULL,      // 13!
    87178291200ULL,     // 14!
    1307674368000ULL,   // 15!
    20922789888000ULL,  // 16!
    355687428096000ULL, // 17!
    6402373705728000ULL,// 18!
    121645100408832000ULL, // 19!
    2432902008176640000ULL // 20!
};
#define MAX_CACHED_FACTORIAL 20

// 安全阶乘函数
static double safe_factorial(int n) {
    if (n <= MAX_CACHED_FACTORIAL) {
        return (double)factorial_cache[n];
    }
    
    // 对于大数，使用double并警告
    double result = (double)factorial_cache[MAX_CACHED_FACTORIAL];
    for (int i = MAX_CACHED_FACTORIAL + 1; i <= n; i++) {
        result *= (double)i;
    }
    return result;
}

// 计算系数
static double compute_coeff(int n, int m, int k) {
    int ma = abs(m);
    int half_sum = (n + ma) / 2;
    int half_diff = (n - ma) / 2;
    
    double fact_nk = safe_factorial(n - k);
    double fact_k = safe_factorial(k);
    double fact_half_sum_k = safe_factorial(half_sum - k);
    double fact_half_diff_k = safe_factorial(half_diff - k);
    
    double coeff = fact_nk / (fact_k * fact_half_sum_k * fact_half_diff_k);
    
    // 处理符号 (-1)^k
    if (k % 2 == 1) {
        coeff = -coeff;
    }
    
    return coeff;
}

// 计算角度项及其导数
static void compute_angular(int m, double x, double y, 
                           double *A, double *dAx, double *dAy) {
    int ma = abs(m);
    
    if (ma == 0) {
        *A = 1.0;
        *dAx = 0.0;
        *dAy = 0.0;
        return;
    }
    
    // 计算 z = x + i*y 的幂
    double z_real = x;
    double z_imag = y;
    
    // z^ma
    double pow_real = 1.0;
    double pow_imag = 0.0;
    
    // z^(ma-1)
    double powm1_real = 1.0;
    double powm1_imag = 0.0;
    
    // 快速幂计算
    if (ma == 1) {
        pow_real = z_real;
        pow_imag = z_imag;
        powm1_real = 1.0;
        powm1_imag = 0.0;
    } else if (ma == 2) {
        // z^2
        pow_real = z_real * z_real - z_imag * z_imag;
        pow_imag = 2.0 * z_real * z_imag;
        // z^1
        powm1_real = z_real;
        powm1_imag = z_imag;
    } else if (ma == 3) {
        // z^3
        double z2_real = z_real * z_real - z_imag * z_imag;
        double z2_imag = 2.0 * z_real * z_imag;
        pow_real = z2_real * z_real - z2_imag * z_imag;
        pow_imag = z2_real * z_imag + z2_imag * z_real;
        // z^2
        powm1_real = z2_real;
        powm1_imag = z2_imag;
    } else {
        // 一般情况：迭代计算
        pow_real = z_real;
        pow_imag = z_imag;
        for (int i = 2; i <= ma; i++) {
            double temp_real = pow_real * z_real - pow_imag * z_imag;
            double temp_imag = pow_real * z_imag + pow_imag * z_real;
            pow_real = temp_real;
            pow_imag = temp_imag;
        }
        
        // 计算 z^(ma-1)
        powm1_real = 1.0;
        powm1_imag = 0.0;
        if (ma - 1 == 1) {
            powm1_real = z_real;
            powm1_imag = z_imag;
        } else {
            for (int i = 1; i <= ma-1; i++) {
                double temp_real = powm1_real * z_real - powm1_imag * z_imag;
                double temp_imag = powm1_real * z_imag + powm1_imag * z_real;
                powm1_real = temp_real;
                powm1_imag = temp_imag;
            }
        }
    }
    
    if (m > 0) {
        // 使用实部
        *A = pow_real;
        *dAx = (double)ma * powm1_real;
        *dAy = -(double)ma * powm1_imag;
    } else {
        // 使用虚部
        *A = pow_imag;
        *dAx = (double)ma * powm1_imag;
        *dAy = (double)ma * powm1_real;
    }
}

// MEX入口函数
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // 检查输入参数
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("Zernike:InvalidInput", 
                         "Four inputs required: n, m, x, y");
    }
    
    if (nlhs != 3) {
        mexErrMsgIdAndTxt("Zernike:InvalidOutput", 
                         "Three outputs required: Z, dZdx, dZdy");
    }
    
    // 获取输入
    int n = (int)mxGetScalar(prhs[0]);
    int m = (int)mxGetScalar(prhs[1]);
    
    const mxArray *x_arr = prhs[2];
    const mxArray *y_arr = prhs[3];
    
    if (!mxIsDouble(x_arr) || !mxIsDouble(y_arr)) {
        mexErrMsgIdAndTxt("Zernike:InvalidType", 
                         "x and y must be double arrays");
    }
    
    // 检查尺寸匹配
    int ndims_x = mxGetNumberOfDimensions(x_arr);
    int ndims_y = mxGetNumberOfDimensions(y_arr);
    if (ndims_x != ndims_y) {
        mexErrMsgIdAndTxt("Zernike:DimensionMismatch", 
                         "x and y must have same dimensions");
    }
    
    const mwSize *dims_x = mxGetDimensions(x_arr);
    const mwSize *dims_y = mxGetDimensions(y_arr);
    int i;
    for (i = 0; i < ndims_x; i++) {
        if (dims_x[i] != dims_y[i]) {
            mexErrMsgIdAndTxt("Zernike:DimensionMismatch", 
                             "x and y must have same dimensions");
        }
    }
    
    // 获取数据指针 - 使用旧版API
    double *x = mxGetPr(x_arr);
    double *y = mxGetPr(y_arr);
    
    // 创建输出数组
    plhs[0] = mxCreateNumericArray(ndims_x, dims_x, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericArray(ndims_x, dims_x, mxDOUBLE_CLASS, mxREAL);
    plhs[2] = mxCreateNumericArray(ndims_x, dims_x, mxDOUBLE_CLASS, mxREAL);
    
    if (plhs[0] == NULL || plhs[1] == NULL || plhs[2] == NULL) {
        mexErrMsgIdAndTxt("Zernike:MemoryError", "Failed to create output arrays");
    }
    
    double *Z = mxGetPr(plhs[0]);
    double *dZdx = mxGetPr(plhs[1]);
    double *dZdy = mxGetPr(plhs[2]);
    
    // 获取元素总数
    mwSize total_elements = 1;
    for (i = 0; i < ndims_x; i++) {
        total_elements *= dims_x[i];
    }
    
    // 参数检查
    int ma = abs(m);
    if ((n - ma) % 2 != 0 || n < ma) {
        mexErrMsgIdAndTxt("Zernike:InvalidMode", 
                         "Invalid Zernike mode: n and m must have same parity and n >= |m|");
    }
    
    // 主循环
    int max_k = (n - ma) / 2;
    
    // 预计算所有系数
    double *coeffs = (double*)mxMalloc((max_k + 1) * sizeof(double));
    if (coeffs == NULL) {
        mexErrMsgIdAndTxt("Zernike:MemoryError", "Out of memory");
    }
    
    int k;
    for (k = 0; k <= max_k; k++) {
        coeffs[k] = compute_coeff(n, m, k);
    }
    
    // 对每个元素进行计算
    mwSize idx;
    for (idx = 0; idx < total_elements; idx++) {
        double x_val = x[idx];
        double y_val = y[idx];
        
        double r2 = x_val * x_val + y_val * y_val;
        
        double Z_val = 0.0;
        double dZdx_val = 0.0;
        double dZdy_val = 0.0;
        
        // 计算角度项（在循环外计算一次，因为不依赖于k）
        double A, dAx, dAy;
        compute_angular(m, x_val, y_val, &A, &dAx, &dAy);
        
        for (k = 0; k <= max_k; k++) {
            int p = n - 2 * k;
            int h = (p - ma) / 2;
            
            // 径向项
            double B, dBx, dBy;
            if (h == 0) {
                B = 1.0;
                dBx = 0.0;
                dBy = 0.0;
            } else {
                double r2_h = pow(r2, (double)h);
                double r2_h_minus_1;
                if (h == 1) {
                    r2_h_minus_1 = 1.0;
                } else {
                    r2_h_minus_1 = pow(r2, (double)(h - 1));
                }
                
                B = r2_h;
                dBx = 2.0 * h * x_val * r2_h_minus_1;
                dBy = 2.0 * h * y_val * r2_h_minus_1;
            }
            
            double coeff = coeffs[k];
            
            // 累加
            Z_val += coeff * A * B;
            dZdx_val += coeff * (dAx * B + A * dBx);
            dZdy_val += coeff * (dAy * B + A * dBy);
        }
        
        Z[idx] = Z_val;
        dZdx[idx] = dZdx_val;
        dZdy[idx] = dZdy_val;
    }
    
    mxFree(coeffs);
}
