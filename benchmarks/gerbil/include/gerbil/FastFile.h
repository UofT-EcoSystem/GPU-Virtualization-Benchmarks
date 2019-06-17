/*********************************************************************************
Copyright (c) 2016 Marius Erbert, Steffen Rechner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*********************************************************************************/

#ifndef FASTFILE_H_
#define FASTFILE_H_

#include <boost/filesystem.hpp>
#include "types.h"

namespace bfs = boost::filesystem;

namespace gerbil {

class FastFile {
	bfs::path _path;
	TFileType _type;
	TFileCompr _compr;
	uint64 _size;
	void init();
public:
	FastFile(const bfs::path &path);
	~FastFile();

	const bfs::path& getPath() const;
	const TFileCompr& getCompr() const;
	const TFileType& getType() const;
	const uint64& getSize() const;
};

}

#endif /* FASTFILE_H_ */
