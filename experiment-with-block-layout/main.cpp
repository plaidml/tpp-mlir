#include <iostream>

using namespace std;

// MNmn += MKmk * NKkn

#define N 16
#define M 6
#define K 8
#define m 2
#define n 2
#define k 2

int main() {
  int C[M][N];
  int CAFTER[M][N];
  int A[M][K];
  int B[K][N];

  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      C[i][j] = 0;

  cout << "C:\n";
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      cout << C[i][j] << " ";
    }
    cout << "\n";
  }

  int cnt = 1;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < K; j++)
      A[i][j] = cnt++;

  cout << "A:\n";
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      cout << A[i][j] << " ";
    }
    cout << "\n";
  }

  for (int i = 0; i < K; i++)
    for (int j = 0; j < N; j++)
      B[i][j] = cnt++;

  cout << "B:\n";
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      cout << B[i][j] << " ";
    }
    cout << "\n";
  }

  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      for (int r = 0; r < K; r++)
        C[i][j] += A[i][r] * B[r][j];

  cout << "GEMM:\n";
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      cout << C[i][j] << " ";
    }
    cout << "\n";
  }

  // MNmn = MKmk * NKkn
  int CBM = M/m;
  int CBN = N/n;
  int BBK = K/k;
  int BBN = N/n;
  int ABM = M/m;
  int ABK = K/k;

  int CBLOCK[CBM][CBN][m][n];
  for (int n1 = 0; n1 < CBM; n1++)
    for (int c1 = 0; c1 < CBN; c1++)
      for (int n2 = 0; n2 < m; n2++)   
        for (int c2 = 0; c2 < n; c2++)
          CBLOCK[n1][c1][n2][c2] = 0;

  int ABLOCK[ABM][ABK][m][k];
  for (int n1 = 0; n1 < ABM; n1++)
    for (int c1 = 0; c1 < ABK; c1++)
      for (int n2 = 0; n2 < m; n2++)
        for (int c2 = 0; c2 < k; c2++)
          ABLOCK[n1][c1][n2][c2] = A[n1*m+n2][c1*k+c2];

  cout << "BLOCKEDA:\n";
  for (int n1 = 0; n1 < ABM; n1++) {
    for (int c1 = 0; c1 < ABK; c1++) {
      for (int n2 = 0; n2 < m; n2++) {
        for (int c2 = 0; c2 < k; c2++) {
          cout << ABLOCK[n1][c1][n2][c2] << " ";
        }
        cout << "\n";
      }
      cout << "\n";
    }
  }
  
  int BBLOCK[BBK][BBN][k][n];
  for (int n1 = 0; n1 < BBK; n1++)
    for (int c1 = 0; c1 < BBN; c1++)
      for (int n2 = 0; n2 < k; n2++)
        for (int c2 = 0; c2 < n; c2++)
          BBLOCK[n1][c1][n2][c2] = B[n1*k+n2][c1*n+c2];

  cout << "BLOCKEDB:\n";
  for (int n1 = 0; n1 < BBK; n1++) {
    for (int c1 = 0; c1 < BBN; c1++) {
      for (int n2 = 0; n2 < k; n2++) {
        for (int c2 = 0; c2 < n; c2++) {
          cout << BBLOCK[n1][c1][n2][c2] << " ";
        }
        cout << "\n";
      }
      cout << "\n";
    }
  }

  cout << "block C on M: " << CBM << "\n";
  cout << "block C on N: " << CBN << "\n";
  cout << "block A on K: " << ABK << "\n";
  cout << "m: " << m << "\n";
  cout << "n: " << n << "\n";
  cout << "k: " << k << "\n";

  for (int p1 = 0; p1 < CBM; p1++)
    for (int p2 = 0; p2 < CBN; p2++)
      for (int r1 = 0; r1 < ABK; r1++)
        for (int p3 = 0; p3 < m; p3++)
          for (int p4 = 0; p4 < n; p4++)
            for (int r2 = 0; r2 < k; r2++)
              CBLOCK[p1][p2][p3][p4] += ABLOCK[p1][r1][p3][r2] * BBLOCK[r1][p2][r2][p4];

  for (int n1 = 0; n1 < CBM; n1++)
    for (int c1 = 0; c1 < CBN; c1++)
      for (int n2 = 0; n2 < m; n2++)
        for (int c2 = 0; c2 < n; c2++)
          CAFTER[n1*m+n2][c1*n+c2] = CBLOCK[n1][c1][n2][c2];

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      cout << CAFTER[i][j] << " ";
    }
    cout << "\n";
  }

  return 0;
}
