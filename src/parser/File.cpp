#include "ast_opt/parser/File.h"

namespace stork {

File::~File() {
  if (_fp) {
    fclose(_fp);
  }
}

File::File(const char *path) :
    _fp(fopen(path, "rt")) {
  if (!_fp) {
    throw stork::file_not_found(std::string("'") + path + "' not found");
  }
}

int File::operator()() {
  return fgetc(_fp);
}

}
