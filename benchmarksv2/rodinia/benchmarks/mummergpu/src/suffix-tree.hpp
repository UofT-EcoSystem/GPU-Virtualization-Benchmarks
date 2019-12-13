#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <sys/time.h>
#include <list>
#include <map>
#include <vector>
#include <queue>
#include <cstring>

#include <sys/stat.h>
#include <fcntl.h>
#include <sys/types.h>
#include <errno.h>
#include <assert.h>
#include <stdint.h> 
#include <unistd.h>

#define ulong4 uint32_t
#define uint4 uint32_t
#define int2 int32_t

#define MPOOL 0

#include "mummergpu.hpp"
#include "PoolMalloc.hpp"

using namespace std;

class EventTime_t
{
public:
  /// Constructor, starts the stopwatch
  EventTime_t()
  {
    start();
    memset(&m_end, 0, sizeof(struct timeval));
  }


  /// Explicitly restart the stopwatch
  void start()
  {
    gettimeofday(&m_start, NULL);
  }


  /// Explicitly stop the stopwatch
  void stop()
  {
    gettimeofday(&m_end, NULL);
  }


  /// Return the duration in seconds
  double duration()
  {
    if ((m_end.tv_sec == 0) && (m_end.tv_usec == 0)) { stop(); }
    return ((m_end.tv_sec - m_start.tv_sec)*1000000.0 + (m_end.tv_usec - m_start.tv_usec)) / 1e6;
  }


  /** \brief Pretty-print the duration in seconds.
   ** If stop() has not already been called, uses the current time as the end
   ** time.
   ** \param format Controls if time should be enclosed in [ ]
   ** \param precision Controls number of digits past decimal pt
   **/
  std::string str(bool format = true,
                  int precision=2)
  {
    double r = duration();

    char buffer[1024];
    sprintf(buffer, "%0.*f", precision, r);

    if (format)
    {
      string s("[");
      s += buffer;
      s += "s]";
      return s;
    }

    return buffer;
  }


private:
  /// Start time
  struct timeval m_start;

  /// End time
  struct timeval m_end;
};

// A node in the suffix tree
class SuffixNode
{
public:
  static int s_nodecount;

#ifdef MPOOL
  void *operator new( size_t num_bytes, PoolMalloc_t *mem)
  {
    return mem->pmalloc(num_bytes);
  }
#endif

  SuffixNode(int s, int e, int leafid, SuffixNode * p, SuffixNode * x);

  ~SuffixNode();

  int id();

  void setPrintParent(int min_match_len);

  bool isLeaf();

  const char * str(const char * refstr);

  int len(int i=-1);

  int depth();

  ostream & printLabel(ostream & os, const char * refstr);
  ostream & printNodeLabel(ostream & os);
  ostream & printEdgeLabel(ostream & os, const char * refstr);

  int setNumLeaves();

  int  m_start;                         // start pos in string
  int  m_end;                           // end pos in string
  int  m_nodeid;                        // the id for this node
  int  m_leafid;                        // For leafs, the start position of the suffix in the string
  int  m_depth;                         // string depth to me
  int  m_numleaves;                     // number of leaves below me
  SuffixNode * m_children [basecount];  // children nodes
  SuffixNode * m_parent;                // parent node
  SuffixNode * m_suffix;                // suffixlink
  SuffixNode * m_printParent;           // where to start printing

#if VERIFY
  string m_pathstring;                  // string of path to node
#endif
};


// Encapsulate the tree with some helper functions
class SuffixTree
{
public:
  SuffixTree(const char * s) : m_string(s)
  {
    m_strlen = strlen(s);
#ifdef MPOOL
    m_root = new (&m_pool) SuffixNode(0,0,0,NULL,NULL); // whole tree
#else
    m_root = new SuffixNode(0,0,0,NULL,NULL); // whole tree
#endif
    m_root->m_suffix = m_root;
  }

  ~SuffixTree()
  {
#ifdef MPOOL
#else
   delete m_root;
#endif
  }

  SuffixNode * m_root;
  const char * m_string;
  int m_strlen;

#ifdef MPOOL
  PoolMalloc_t m_pool;
#endif

  // Print a node for dot
  void printNodeDot(SuffixNode * node, ostream & dfile);

  // Print the whole tree for dot
  void printDot(const char * dotfilename);

  // Print a node in text format
  void printNodeText(ostream & out, SuffixNode * n, int depth);

  // Print the tree in Text
  void printText(ostream & out);

  // Print the tree as list of sorted suffixes
  void printTreeSorted(ostream & out, SuffixNode * node, const string & pathstring);

  void printTreeFlat(ostream & out);

  void printNodeFlat(ostream & out, SuffixNode * node);

#if VERIFY
  void setNodePath(SuffixNode * node, const string & parentString);
  int verifyNodeSuffixLinks(SuffixNode * node, int & linkcount);
  void verifySuffixLinks();
#endif

  void buildUkkonen();
};
