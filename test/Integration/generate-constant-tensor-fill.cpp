#include <iostream>

int main() {
  int row = 1;
  int col = 1;

  int rows = 32;
  int cols = 16;

  while (row <= rows) {
    std::cout << "[ ";
    while (col <= cols) {
      if (col == cols)
        std::cout << col << "." << row << " ";
      else
        std::cout << col << "." << row << ", ";
      col++;
    }
    if (row == rows)
      std::cout << "]\n";
    else
      std::cout << "],\n";
    row++;
    col = 1;
  }

  return 0;
}
