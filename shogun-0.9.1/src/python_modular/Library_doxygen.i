
// File: index.xml

// File: classshogun_1_1CArray.xml
%feature("docstring") shogun::CArray "

Template class Array implements a dense one dimensional array.

Note that depending on compile options everything will be inlined,
such that this is as high performance array implementation without
error checking.

C++ includes: Array.h ";

%feature("docstring")  shogun::CArray::CArray "

constructor

Parameters:
-----------

initial_size:  initial size of array ";

%feature("docstring")  shogun::CArray::CArray "

constructor

Parameters:
-----------

p_array:  another array

p_array_size:  size of another array

p_free_array:  if array must be freed

p_copy_array:  if array must be copied ";

%feature("docstring")  shogun::CArray::CArray "

constructor

Parameters:
-----------

p_array:  another array

p_array_size:  size of another array ";

%feature("docstring")  shogun::CArray::~CArray "";

%feature("docstring")  shogun::CArray::get_name "

get name

name ";

%feature("docstring")  shogun::CArray::set_name "

set name

Parameters:
-----------

p_name:  new name ";

%feature("docstring")  shogun::CArray::get_array_size "

get array size (including granularity buffer)

total array size ";

%feature("docstring")  shogun::CArray::get_dim1 "

get array size (including granularity buffer)

total array size ";

%feature("docstring")  shogun::CArray::zero "

zero array ";

%feature("docstring")  shogun::CArray::set_const "

set array with a constant ";

%feature("docstring")  shogun::CArray::get_element "

get array element at index

Parameters:
-----------

index:  index

array element at index ";

%feature("docstring")  shogun::CArray::set_element "

set array element at index 'index' return false in case of trouble

Parameters:
-----------

p_element:  array element to set

index:  index

if setting was successful ";

%feature("docstring")  shogun::CArray::element "

get element at given index

Parameters:
-----------

idx1:  index

element at given index ";

%feature("docstring")  shogun::CArray::element "

get element at given index

Parameters:
-----------

index:  index

element at given index ";

%feature("docstring")  shogun::CArray::element "

get element of given array at given index

Parameters:
-----------

p_array:  another array

index:  index

element of given array at given index ";

%feature("docstring")  shogun::CArray::resize_array "

resize array

Parameters:
-----------

n:  new size

if resizing was successful ";

%feature("docstring")  shogun::CArray::get_array "

call get_array just before messing with it DO NOT call any
[],resize/delete functions after get_array(), the pointer may become
invalid!

the array ";

%feature("docstring")  shogun::CArray::set_array "

set the array pointer and free previously allocated memory

Parameters:
-----------

p_array:  another array

p_array_size:  size of another array

p_free_array:  if array must be freed

copy_array:  if array must be copied ";

%feature("docstring")  shogun::CArray::set_array "

set the array pointer and free previously allocated memory

Parameters:
-----------

p_array:  another array

p_array_size:  size of another array ";

%feature("docstring")  shogun::CArray::clear_array "

clear the array (with zeros) ";

%feature("docstring")  shogun::CArray::display_size "

display array size ";

%feature("docstring")  shogun::CArray::display_array "

display array ";


// File: classshogun_1_1CArray2.xml
%feature("docstring") shogun::CArray2 "

Template class Array2 implements a dense two dimensional array.

Note that depending on compile options everything will be inlined,
such that this is as high performance 2d-array implementation without
error checking.

C++ includes: Array2.h ";

%feature("docstring")  shogun::CArray2::CArray2 "

constructor

Parameters:
-----------

dim1:  dimension 1

dim2:  dimension 2 ";

%feature("docstring")  shogun::CArray2::CArray2 "

constructor

Parameters:
-----------

p_array:  another array

dim1:  dimension 1

dim2:  dimension 2

p_free_array:  if array must be freed

p_copy_array:  if array must be copied ";

%feature("docstring")  shogun::CArray2::CArray2 "

constructor

Parameters:
-----------

p_array:  another array

dim1:  dimension 1

dim2:  dimension 2 ";

%feature("docstring")  shogun::CArray2::~CArray2 "";

%feature("docstring")  shogun::CArray2::get_array_size "

return total array size (including granularity buffer)

Parameters:
-----------

dim1:  dimension 1 will be stored here

dim2:  dimension 2 will be stored here ";

%feature("docstring")  shogun::CArray2::get_dim1 "

get dimension 1

dimension 1 ";

%feature("docstring")  shogun::CArray2::get_dim2 "

get dimension 2

dimension 2 ";

%feature("docstring")  shogun::CArray2::zero "

zero array ";

%feature("docstring")  shogun::CArray2::set_const "

set array with a constant ";

%feature("docstring")  shogun::CArray2::get_array "

get the array call get_array just before messing with it DO NOT call
any [],resize/delete functions after get_array(), the pointer may
become invalid !

the array ";

%feature("docstring")  shogun::CArray2::set_name "

set array's name

Parameters:
-----------

p_name:  new name ";

%feature("docstring")  shogun::CArray2::set_array "

set the array pointer and free previously allocated memory

Parameters:
-----------

p_array:  another array

dim1:  dimension 1

dim2:  dimensino 2

p_free_array:  if array must be freed

copy_array:  if array must be copied ";

%feature("docstring")  shogun::CArray2::resize_array "

resize array

Parameters:
-----------

dim1:  new dimension 1

dim2:  new dimension 2

if resizing was successful ";

%feature("docstring")  shogun::CArray2::get_element "

get array element at index

Parameters:
-----------

idx1:  index 1

idx2:  index 2

array element at index ";

%feature("docstring")  shogun::CArray2::set_element "

set array element at index 'index'

Parameters:
-----------

p_element:  array element

idx1:  index 1

idx2:  index 2

if setting was successful ";

%feature("docstring")  shogun::CArray2::element "

get array element at index

Parameters:
-----------

idx1:  index 1

idx2:  index 2

array element at index ";

%feature("docstring")  shogun::CArray2::element "

get array element at index

Parameters:
-----------

idx1:  index 1

idx2:  index 2

array element at index ";

%feature("docstring")  shogun::CArray2::element "

get element of given array at given index

Parameters:
-----------

p_array:  another array

idx1:  index 1

idx2:  index 2

element of given array at given index ";

%feature("docstring")  shogun::CArray2::element "

get element of given array at given index

Parameters:
-----------

p_array:  another array

idx1:  index 1

idx2:  index 2

p_dim1_size:  size of dimension 1

element of given array at given index ";

%feature("docstring")  shogun::CArray2::display_array "

display array ";

%feature("docstring")  shogun::CArray2::display_size "

display array size ";

%feature("docstring")  shogun::CArray2::get_name "

object name ";


// File: classshogun_1_1CArray3.xml
%feature("docstring") shogun::CArray3 "

Template class Array3 implements a dense three dimensional array.

Note that depending on compile options everything will be inlined,
such that this is as high performance 3d-array implementation without
error checking.

C++ includes: Array3.h ";

%feature("docstring")  shogun::CArray3::CArray3 "

default constructor ";

%feature("docstring")  shogun::CArray3::CArray3 "

constructor

Parameters:
-----------

dim1:  dimension 1

dim2:  dimension 2

dim3:  dimension 3 ";

%feature("docstring")  shogun::CArray3::CArray3 "

constructor

Parameters:
-----------

p_array:  another array

dim1:  dimension 1

dim2:  dimension 2

dim3:  dimension 3

p_free_array:  if array must be freed

p_copy_array:  if array must be copied ";

%feature("docstring")  shogun::CArray3::CArray3 "

constructor

Parameters:
-----------

p_array:  another array

dim1:  dimension 1

dim2:  dimension 2

dim3:  dimension 3 ";

%feature("docstring")  shogun::CArray3::~CArray3 "";

%feature("docstring")  shogun::CArray3::set_name "

set array's name

Parameters:
-----------

p_name:  new name ";

%feature("docstring")  shogun::CArray3::get_array_size "

return total array size (including granularity buffer)

Parameters:
-----------

dim1:  dimension 1 will be stored here

dim2:  dimension 2 will be stored here

dim3:  dimension 3 will be stored here ";

%feature("docstring")  shogun::CArray3::get_dim1 "

get dimension 1

dimension 1 ";

%feature("docstring")  shogun::CArray3::get_dim2 "

get dimension 2

dimension 2 ";

%feature("docstring")  shogun::CArray3::get_dim3 "

get dimension 3

dimension 3 ";

%feature("docstring")  shogun::CArray3::zero "

zero array ";

%feature("docstring")  shogun::CArray3::set_const "

set array with a constant ";

%feature("docstring")  shogun::CArray3::get_array "

get the array call get_array just before messing with it DO NOT call
any [],resize/delete functions after get_array(), the pointer may
become invalid !

the array ";

%feature("docstring")  shogun::CArray3::set_array "

set the array pointer and free previously allocated memory

Parameters:
-----------

p_array:  another array

dim1:  dimension 1

dim2:  dimensino 2

dim3:  dimensino 3

p_free_array:  if array must be freed

copy_array:  if array must be copied ";

%feature("docstring")  shogun::CArray3::resize_array "

resize array

Parameters:
-----------

dim1:  new dimension 1

dim2:  new dimension 2

dim3:  new dimension 3

if resizing was successful ";

%feature("docstring")  shogun::CArray3::get_element "

get array element at index

Parameters:
-----------

idx1:  index 1

idx2:  index 2

idx3:  index 3

array element at index ";

%feature("docstring")  shogun::CArray3::set_element "

set array element at index 'index'

Parameters:
-----------

p_element:  array element

idx1:  index 1

idx2:  index 2

idx3:  index 3

if setting was successful ";

%feature("docstring")  shogun::CArray3::element "

get array element at index

Parameters:
-----------

idx1:  index 1

idx2:  index 2

idx3:  index 3

array element at index ";

%feature("docstring")  shogun::CArray3::element "

get array element at index

Parameters:
-----------

idx1:  index 1

idx2:  index 2

idx3:  index 3

array element at index ";

%feature("docstring")  shogun::CArray3::element "

get element of given array at given index

Parameters:
-----------

p_array:  another array

idx1:  index 1

idx2:  index 2

idx3:  index 3

array element at index ";

%feature("docstring")  shogun::CArray3::element "

get element of given array at given index

Parameters:
-----------

p_array:  another array

idx1:  index 1

idx2:  index 2

idx3:  index 3

p_dim1_size:  size of dimension 1

p_dim2_size:  size of dimension 2

element of given array at given index ";

%feature("docstring")  shogun::CArray3::display_size "

display array size ";

%feature("docstring")  shogun::CArray3::display_array "

display array ";

%feature("docstring")  shogun::CArray3::get_name "

object name ";


// File: classshogun_1_1CBinaryStream.xml
%feature("docstring") shogun::CBinaryStream "

memory mapped emulation via binary streams (files)

Implements memory mapped file emulation ( See:   CMemoryMappedFile)
via standard file operations like fseek, fread etc

C++ includes: BinaryStream.h ";

%feature("docstring")  shogun::CBinaryStream::CBinaryStream "

default constructor ";

%feature("docstring")  shogun::CBinaryStream::CBinaryStream "

constructor

open a file for read mode

Parameters:
-----------

fname:  name of file, zero terminated string

flag:  determines read or read write mode (currently only 'r' is
supported) ";

%feature("docstring")  shogun::CBinaryStream::CBinaryStream "

copy constructor

Parameters:
-----------

bs:  binary stream to copy from ";

%feature("docstring")  shogun::CBinaryStream::~CBinaryStream "

destructor ";

%feature("docstring")  shogun::CBinaryStream::open_stream "

open file stream

Parameters:
-----------

fname:  file name

flag:  flags \"r\" for reading etc ";

%feature("docstring")  shogun::CBinaryStream::close_stream "

close a file stream ";

%feature("docstring")  shogun::CBinaryStream::get_length "

get the number of objects of type T cointained in the file

length of file ";

%feature("docstring")  shogun::CBinaryStream::get_size "

get the size of the file in bytes

size of file in bytes ";

%feature("docstring")  shogun::CBinaryStream::get_line "

get next line from file

The returned line may be modfied in case the file was opened
read/write. It is otherwise read-only.

Parameters:
-----------

len:  length of line (returned via reference)

offs:  offset to be passed for reading next line, should be 0
initially (returned via reference)

line (NOT ZERO TERMINATED) ";

%feature("docstring")  shogun::CBinaryStream::get_num_lines "

count the number of lines in a file

number of lines ";

%feature("docstring")  shogun::CBinaryStream::pre_buffer "

read num elements starting from index into buffer

Parameters:
-----------

buffer:  buffer that has to be at least num elements long

index:  index into file starting from which elements are read

num:  number of elements to be read ";

%feature("docstring")  shogun::CBinaryStream::read_next "

read next

next element ";

%feature("docstring")  shogun::CBinaryStream::get_name "

object name ";


// File: classshogun_1_1CBitString.xml
%feature("docstring") shogun::CBitString "

a string class embedding a string in a compact bit representation

especially useful to compactly represent genomic DNA

(or any other string of small alphabet size)

C++ includes: BitString.h ";

%feature("docstring")  shogun::CBitString::CBitString "

default constructor

creates an empty Bitstring

Parameters:
-----------

alpha:  Alphabet

width:  return this many bits upon str[idx] access operations ";

%feature("docstring")  shogun::CBitString::~CBitString "

destructor ";

%feature("docstring")  shogun::CBitString::cleanup "

free up memory ";

%feature("docstring")  shogun::CBitString::obtain_from_char "

convert string of length len into bit sequence

Parameters:
-----------

str:  string

len:  length of string in bits ";

%feature("docstring")  shogun::CBitString::load_fasta_file "

load fasta file as bit string

Parameters:
-----------

fname:  filename to load from

ignore_invalid:  if set to true, characters other than A,C,G,T are
converted to A

if loading was successful ";

%feature("docstring")  shogun::CBitString::set_string "

set string of length len embedded in a uint64_t sequence

Parameters:
-----------

str:  string

len:  length of string in bits ";

%feature("docstring")  shogun::CBitString::create "

creates string of all zeros of len bits

Parameters:
-----------

len:  length of string in bits ";

%feature("docstring")  shogun::CBitString::set_binary_word "

set a binary word

Parameters:
-----------

word:  16 bit word to be set

index:  word based index ";

%feature("docstring")  shogun::CBitString::get_length "

length of the string in bits ";

%feature("docstring")  shogun::CBitString::get_name "

object name ";


// File: classshogun_1_1CCache.xml
%feature("docstring") shogun::CCache "

Template class Cache implements a simple cache.

When the cache is full -- elements that are least used are freed from
the cache. Thus for the cache to be effective one should not visit
loop over objects, i.e. visit elements in order 0...num_elements (with
num_elements >> the maximal number of entries in cache)

C++ includes: Cache.h ";

%feature("docstring")  shogun::CCache::CCache "

constructor

create a cache in which num_entries objects can be cached whose lookup
table of sizeof(int64_t)*num_entries must fit into memory

Parameters:
-----------

cache_size:  cache size in Megabytes

obj_size:  object size

num_entries:  number of cached objects ";

%feature("docstring")  shogun::CCache::~CCache "";

%feature("docstring")  shogun::CCache::is_cached "

checks if an object is cached

Parameters:
-----------

number:  number of object to check for

if an object is cached ";

%feature("docstring")  shogun::CCache::lock_entry "

lock and get a cache entry

Parameters:
-----------

number:  number of object to lock and get

cache entry or NULL when not cached ";

%feature("docstring")  shogun::CCache::unlock_entry "

unlock a cache entry

Parameters:
-----------

number:  number of object to unlock ";

%feature("docstring")  shogun::CCache::set_entry "

returns the address of a free cache entry to where the data of size
obj_size has to be written

Parameters:
-----------

number:  number of object to unlock

address of a free cache entry ";

%feature("docstring")  shogun::CCache::get_name "

object name ";


// File: classshogun_1_1CCompressor.xml
%feature("docstring") shogun::CCompressor "

Compression library for compressing and decompressing buffers using
one of the standard compression algorithms, LZO, GZIP, BZIP2 or LZMA.

The general recommendation is to use LZO whenever lightweight
compression is sufficient but high i/o throughputs are needed (at 1/2
the speed of memcpy).

If size is all that matters use LZMA (which especially when
compressing can be very slow though).

Note that besides lzo compression, this library is thread safe.

C++ includes: Compressor.h ";

%feature("docstring")  shogun::CCompressor::CCompressor "

default constructor

Parameters:
-----------

ct:  compression to use: one of UNCOMPRESSED, LZO, GZIP, BZIP2 or LZMA
";

%feature("docstring")  shogun::CCompressor::~CCompressor "

default destructor ";

%feature("docstring")  shogun::CCompressor::compress "

compress data

compresses the buffer uncompressed using the selected compression
algorithm and returns compressed data and its size

Parameters:
-----------

uncompressed:  - uncompressed data to be compressed

uncompressed_size:  - size of the uncompressed data

compressed:  - pointer to hold compressed data (returned)

compressed_size:  - size of compressed data (returned)

level:  - compression level between 1 and 9 ";

%feature("docstring")  shogun::CCompressor::decompress "

decompress data

Decompresses the buffer using the selected compression algorithm to
the memory block specified in uncompressed. Note: Compressed and
uncompressed size must be known prior to calling this function.

Parameters:
-----------

compressed:  - pointer to compressed data

compressed_size:  - size of compressed data

uncompressed:  - pointer to buffer to hold uncompressed data

uncompressed_size:  - size of the uncompressed data ";

%feature("docstring")  shogun::CCompressor::get_name "

object name ";


// File: classshogun_1_1CDynamicArray.xml
%feature("docstring") shogun::CDynamicArray "

Template Dynamic array class that creates an array that can be used
like a list or an array.

It grows and shrinks dynamically, while elements can be accessed via
index. It is performance tuned for simple types like float etc. and
for hi-level objects only stores pointers, which are not automagically
SG_REF'd/deleted.

C++ includes: DynamicArray.h ";

%feature("docstring")  shogun::CDynamicArray::CDynamicArray "

constructor

Parameters:
-----------

p_resize_granularity:  resize granularity ";

%feature("docstring")  shogun::CDynamicArray::~CDynamicArray "";

%feature("docstring")  shogun::CDynamicArray::set_granularity "

set the resize granularity

Parameters:
-----------

g:  new granularity

what has been set (minimum is 128) ";

%feature("docstring")  shogun::CDynamicArray::get_array_size "

get array size (including granularity buffer)

total array size (including granularity buffer) ";

%feature("docstring")  shogun::CDynamicArray::get_num_elements "

get number of elements

number of elements ";

%feature("docstring")  shogun::CDynamicArray::get_element "

get array element at index

(does NOT do bounds checking)

Parameters:
-----------

index:  index

array element at index ";

%feature("docstring")  shogun::CDynamicArray::get_element_safe "

get array element at index

(does bounds checking)

Parameters:
-----------

index:  index

array element at index ";

%feature("docstring")  shogun::CDynamicArray::set_element "

set array element at index

Parameters:
-----------

element:  element to set

index:  index

if setting was successful ";

%feature("docstring")  shogun::CDynamicArray::insert_element "

insert array element at index

Parameters:
-----------

element:  element to insert

index:  index

if setting was successful ";

%feature("docstring")  shogun::CDynamicArray::append_element "

append array element to the end of array

Parameters:
-----------

element:  element to append

if setting was successful ";

%feature("docstring")  shogun::CDynamicArray::find_element "

find first occurence of array element and return its index or -1 if
not available

Parameters:
-----------

element:  element to search for

index of element or -1 ";

%feature("docstring")  shogun::CDynamicArray::delete_element "

delete array element at idx (does not call delete[] or the like)

Parameters:
-----------

idx:  index

if deleting was successful ";

%feature("docstring")  shogun::CDynamicArray::resize_array "

resize the array

Parameters:
-----------

n:  new size

if resizing was successful ";

%feature("docstring")  shogun::CDynamicArray::get_array "

get the array call get_array just before messing with it DO NOT call
any [],resize/delete functions after get_array(), the pointer may
become invalid !

the array ";

%feature("docstring")  shogun::CDynamicArray::set_array "

set the array pointer and free previously allocated memory

Parameters:
-----------

p_array:  new array

p_num_elements:  last element index + 1

array_size:  number of elements in array ";

%feature("docstring")  shogun::CDynamicArray::clear_array "

clear the array (with zeros) ";

%feature("docstring")  shogun::CDynamicArray::get_name "

object name ";


// File: classshogun_1_1CDynInt.xml
%feature("docstring") shogun::CDynInt "

integer type of dynamic size

This object can be used to create huge integers. These integers can be
used directly instead of the usual int32_t etc types since operators
are properly overloaded.

An exampe use would be 512 wide unsigned ints consisting of four
uint64's:

CDynInt<uint64_t, 4> int512;

This data type is mostly used as a (efficient) storage container for
bit-mapped strings. Therefore, currently only comparison, assignment
and bit operations are implemented.

TODO: implement add,mul,div

C++ includes: DynInt.h ";

%feature("docstring")  shogun::CDynInt::CDynInt "

default constructor

creates a DynInt that is all zero. ";

%feature("docstring")  shogun::CDynInt::CDynInt "

constructor (set least significant ``word'')

The least significant word is set, the rest filled with zeros.

Parameters:
-----------

x:  least significant word ";

%feature("docstring")  shogun::CDynInt::CDynInt "

constructor (set least significant ``word'')

The least significant word is set, the rest filled with zeros.

Parameters:
-----------

x:  least significant word ";

%feature("docstring")  shogun::CDynInt::CDynInt "

constructor (set least significant ``word'')

The least significant word is set, the rest filled with zeros.

Parameters:
-----------

x:  least significant word ";

%feature("docstring")  shogun::CDynInt::CDynInt "

constructor (set least significant ``word'')

The least significant word is set, the rest filled with zeros.

Parameters:
-----------

x:  least significant word ";

%feature("docstring")  shogun::CDynInt::CDynInt "

constructor (set least significant ``word'')

The least significant word is set, the rest filled with zeros.

Parameters:
-----------

x:  least significant word ";

%feature("docstring")  shogun::CDynInt::CDynInt "

constructor (set least significant ``word'')

The least significant word is set, the rest filled with zeros.

Parameters:
-----------

x:  least significant word ";

%feature("docstring")  shogun::CDynInt::CDynInt "

constructor (set whole array)

Initialize the DynInt based on an array, which is passed as an
argument.

Parameters:
-----------

x:  array of size sz ";

%feature("docstring")  shogun::CDynInt::CDynInt "

copy constructor ";

%feature("docstring")  shogun::CDynInt::~CDynInt "

destructor ";

%feature("docstring")  shogun::CDynInt::print_hex "

print the current long integer in hex (without carriage return ";

%feature("docstring")  shogun::CDynInt::print_bits "

print the current long integer in bits (without carriage return ";


// File: classshogun_1_1CFile.xml
%feature("docstring") shogun::CFile "

A File access class.

A file consists of a fourcc header then an alternation of a type
header and data or just raw data (simplefile=true). However this
implementation is not complete - the more complex stuff is currently
not implemented.

C++ includes: File.h ";

%feature("docstring")  shogun::CFile::CFile "

constructor

Parameters:
-----------

f:  already opened file ";

%feature("docstring")  shogun::CFile::CFile "

constructor

Parameters:
-----------

fname:  filename to open

rw:  mode, 'r' or 'w'

type:  specifies the datatype used in the file (F_INT,...)

fourcc:  in the case fourcc is 0, type will be ignored and the file is
treated as if it has a header/[typeheader,data]+ else the files header
will be checked to contain the specified fourcc (e.g. 'RFEA') ";

%feature("docstring")  shogun::CFile::~CFile "";

%feature("docstring")  shogun::CFile::parse_first_header "

parse first header - defunct!

Parameters:
-----------

type:  feature type

-1 ";

%feature("docstring")  shogun::CFile::parse_next_header "

parse next header - defunct!

Parameters:
-----------

type:  feature type

-1 ";

%feature("docstring")  shogun::CFile::load_int_data "

load integer data

Parameters:
-----------

target:  loaded data

num:  number of data elements

loaded data ";

%feature("docstring")  shogun::CFile::load_real_data "

load real data

Parameters:
-----------

target:  loaded data

num:  number of data elements

loaded data ";

%feature("docstring")  shogun::CFile::load_shortreal_data "

load shortreal data

Parameters:
-----------

target:  loaded data

num:  number of data elements

loaded data ";

%feature("docstring")  shogun::CFile::load_char_data "

load char data

Parameters:
-----------

target:  loaded data

num:  number of data elements

loaded data ";

%feature("docstring")  shogun::CFile::load_byte_data "

load byte data

Parameters:
-----------

target:  loaded data

num:  number of data elements

loaded data ";

%feature("docstring")  shogun::CFile::load_word_data "

load word data

Parameters:
-----------

target:  loaded data

num:  number of data elements

loaded data ";

%feature("docstring")  shogun::CFile::load_short_data "

load short data

Parameters:
-----------

target:  loaded data

num:  number of data elements

loaded data ";

%feature("docstring")  shogun::CFile::load_data "

load data (templated)

Parameters:
-----------

target:  loaded data

num:  number of data elements

loaded data ";

%feature("docstring")  shogun::CFile::save_data "

save data (templated)

Parameters:
-----------

src:  data to save

num:  number of data elements

whether operation was successful ";

%feature("docstring")  shogun::CFile::save_int_data "

save integer data

Parameters:
-----------

src:  data to save

num:  number of data elements

whether operation was successful ";

%feature("docstring")  shogun::CFile::save_real_data "

save real data

Parameters:
-----------

src:  data to save

num:  number of data elements

whether operation was successful ";

%feature("docstring")  shogun::CFile::save_shortreal_data "

save shortreal data

Parameters:
-----------

src:  data to save

num:  number of data elements

whether operation was successful ";

%feature("docstring")  shogun::CFile::save_char_data "

save char data

Parameters:
-----------

src:  data to save

num:  number of data elements

whether operation was successful ";

%feature("docstring")  shogun::CFile::save_byte_data "

save byte data

Parameters:
-----------

src:  data to save

num:  number of data elements

whether operation was successful ";

%feature("docstring")  shogun::CFile::save_word_data "

save word data

Parameters:
-----------

src:  data to save

num:  number of data elements

whether operation was successful ";

%feature("docstring")  shogun::CFile::save_short_data "

save short data

Parameters:
-----------

src:  data to save

num:  number of data elements

whether operation was successful ";

%feature("docstring")  shogun::CFile::is_ok "

check if status is ok

whether status is ok ";

%feature("docstring")  shogun::CFile::read_real_valued_sparse "

read sparse real valued features in svm light format e.g. -1 1:10.0
2:100.2 1000:1.3 with -1 == (optional) label and dim 1 - value 10.0
dim 2 - value 100.2 dim 1000 - value 1.3

Parameters:
-----------

matrix:  matrix to read into

num_feat:  number of features for each vector

num_vec:  number of vectors in matrix

if reading was successful ";

%feature("docstring")  shogun::CFile::write_real_valued_sparse "

write sparse real valued features in svm light format

Parameters:
-----------

matrix:  matrix to write

num_feat:  number of features for each vector

num_vec:  number of vectros in matrix

if writing was successful ";

%feature("docstring")  shogun::CFile::read_real_valued_dense "

read dense real valued features, simple ascii format e.g. 1.0 1.1 0.2
2.3 3.5 5

a matrix that consists of 3 vectors with each of 2d

Parameters:
-----------

matrix:  matrix to read into

num_feat:  number of features for each vector

num_vec:  number of vectors in matrix

if reading was successful ";

%feature("docstring")  shogun::CFile::write_real_valued_dense "

write dense real valued features, simple ascii format

Parameters:
-----------

matrix:  matrix to write

num_feat:  number of features for each vector

num_vec:  number of vectros in matrix

if writing was successful ";

%feature("docstring")  shogun::CFile::read_char_valued_strings "

read char string features, simple ascii format e.g. foo bar
ACGTACGTATCT

two strings

Parameters:
-----------

strings:  strings to read into

num_str:  number of strings

max_string_len:  length of longest string

if reading was successful ";

%feature("docstring")  shogun::CFile::write_char_valued_strings "

write char string features, simple ascii format

Parameters:
-----------

strings:  strings to write

num_str:  number of strings

if writing was successful ";

%feature("docstring")  shogun::CFile::get_name "

object name ";


// File: classshogun_1_1CGCArray.xml
%feature("docstring") shogun::CGCArray "

Template class GCArray implements a garbage collecting static array.

This array is meant to be used for Shogun Objects (CSGObject) only, as
it deals with garbage collection, i.e. on read and array assignment
the reference count is increased (and decreased on delete and
overwriting elements).

C++ includes: GCArray.h ";

%feature("docstring")  shogun::CGCArray::CGCArray "

Constructor

Parameters:
-----------

sz:  length of array ";

%feature("docstring")  shogun::CGCArray::~CGCArray "

Destructor ";

%feature("docstring")  shogun::CGCArray::set "

write access operator

Parameters:
-----------

element:  - element to write

index:  - index to write to ";

%feature("docstring")  shogun::CGCArray::get "

read only access operator

Parameters:
-----------

index:  index to write to

element element ";

%feature("docstring")  shogun::CGCArray::get_name "

get the name of the object

name of object ";


// File: classshogun_1_1CHash.xml
%feature("docstring") shogun::CHash "

Collection of Hashing Functions.

This class implements a number of hashing functions like crc32, md5
and murmur.

C++ includes: Hash.h ";

%feature("docstring")  shogun::CHash::CHash "

default constructor ";

%feature("docstring")  shogun::CHash::~CHash "

default destructor ";

%feature("docstring")  shogun::CHash::MurmurHash2 "

Murmur Hash2

Parameters:
-----------

data:  data to checksum (needs to be 32bit aligned on some archs)

len:  length in number of bytes

seed:  initial seed

hash ";

%feature("docstring")  shogun::CHash::get_name "

object name ";


// File: classshogun_1_1CIndirectObject.xml
%feature("docstring") shogun::CIndirectObject "

an array class that accesses elements indirectly via an index array.

It does not store the objects itself, but only indices to objects.
This conveniently allows e.g. sorting the array without changing the
order of objects (but only the order of their indices).

C++ includes: IndirectObject.h ";

%feature("docstring")  shogun::CIndirectObject::CIndirectObject "

default constructor (initializes index with -1) ";

%feature("docstring")  shogun::CIndirectObject::CIndirectObject "

constructor

Parameters:
-----------

idx:  index ";


// File: classshogun_1_1CIO.xml
%feature("docstring") shogun::CIO "

Class IO, used to do input output operations throughout shogun.

Any debug or error or progress message is passed through the functions
of this class to be in the end written to the screen. Note that
messages don't have to be written to stdout or stderr, but can be
redirected to a file.

C++ includes: io.h ";

%feature("docstring")  shogun::CIO::CIO "

default constructor ";

%feature("docstring")  shogun::CIO::CIO "

copy constructor ";

%feature("docstring")  shogun::CIO::set_loglevel "

set loglevel

Parameters:
-----------

level:  level of log messages ";

%feature("docstring")  shogun::CIO::get_loglevel "

get loglevel

level of log messages ";

%feature("docstring")  shogun::CIO::get_show_progress "

get show_progress

if progress bar is shown ";

%feature("docstring")  shogun::CIO::get_show_file_and_line "

get show file and line

if file and line should prefix messages ";

%feature("docstring")  shogun::CIO::message "

print a message

optionally prefixed with file name and line number from (use -1 in
line to disable this)

Parameters:
-----------

prio:  message priority

file:  file name from where the message is called

line:  line number from where the message is called

fmt:  format string ";

%feature("docstring")  shogun::CIO::progress "

print progress bar

Parameters:
-----------

current_val:  current value

min_val:  minimum value

max_val:  maximum value

decimals:  decimals

prefix:  message prefix ";

%feature("docstring")  shogun::CIO::absolute_progress "

print absolute progress bar

Parameters:
-----------

current_val:  current value

val:  value

min_val:  minimum value

max_val:  maximum value

decimals:  decimals

prefix:  message prefix ";

%feature("docstring")  shogun::CIO::done "

print 'done' with priority INFO, but only if progress bar is enabled
";

%feature("docstring")  shogun::CIO::not_implemented "

print error message 'not implemented' ";

%feature("docstring")  shogun::CIO::deprecated "

print warning message 'function deprecated' ";

%feature("docstring")  shogun::CIO::buffered_message "

print a buffered message

Parameters:
-----------

prio:  message priority

fmt:  format string ";

%feature("docstring")  shogun::CIO::get_target "

get target

file descriptor for target ";

%feature("docstring")  shogun::CIO::set_target "

set target

Parameters:
-----------

target:  file descriptor for target ";

%feature("docstring")  shogun::CIO::set_target_to_stderr "

set target to stderr ";

%feature("docstring")  shogun::CIO::set_target_to_stdout "

set target to stdout ";

%feature("docstring")  shogun::CIO::enable_progress "

enable progress bar ";

%feature("docstring")  shogun::CIO::disable_progress "

disable progress bar ";

%feature("docstring")  shogun::CIO::enable_file_and_line "

enable displaying of file and line when printing messages ";

%feature("docstring")  shogun::CIO::disable_file_and_line "

disable displaying of file and line when printing messages ";

%feature("docstring")  shogun::CIO::ref "

increase reference counter

reference count ";

%feature("docstring")  shogun::CIO::ref_count "

display reference counter

reference count ";

%feature("docstring")  shogun::CIO::unref "

decrement reference counter and deallocate object if refcount is zero
before or after decrementing it

reference count ";

%feature("docstring")  shogun::CIO::get_name "

object name ";


// File: classshogun_1_1CList.xml
%feature("docstring") shogun::CList "

Class List implements a doubly connected list for low-level-objects.

For higher level objects pointers should be used. The list supports
calling delete() of an object that is to be removed from the list.

C++ includes: List.h ";

/*  thread safe list access functions  */

/*

*/

%feature("docstring")  shogun::CList::get_first_element "

go to first element in list and return it

Parameters:
-----------

p_current:  current list element

first element in list or NULL if list is empty ";

%feature("docstring")  shogun::CList::get_last_element "

go to last element in list and return it

Parameters:
-----------

p_current:  current list element

last element in list or NULL if list is empty ";

%feature("docstring")  shogun::CList::get_next_element "

go to next element in list and return it

Parameters:
-----------

p_current:  current list element

next element in list or NULL if list is empty ";

%feature("docstring")  shogun::CList::get_previous_element "

go to previous element in list and return it

Parameters:
-----------

p_current:  current list element

previous element in list or NULL if list is empty ";

%feature("docstring")  shogun::CList::get_current_element "

get current element in list

Parameters:
-----------

p_current:  current list element

current element in list or NULL if not available ";

%feature("docstring")  shogun::CList::CList "

constructor

Parameters:
-----------

p_delete_data:  if data shall be deleted ";

%feature("docstring")  shogun::CList::~CList "";

%feature("docstring")  shogun::CList::get_num_elements "

get number of elements in list

number of elements in list ";

%feature("docstring")  shogun::CList::get_first_element "

go to first element in list and return it

first element in list or NULL if list is empty ";

%feature("docstring")  shogun::CList::get_last_element "

go to last element in list and return it

last element in list or NULL if list is empty ";

%feature("docstring")  shogun::CList::get_next_element "

go to next element in list and return it

next element in list or NULL if list is empty ";

%feature("docstring")  shogun::CList::get_previous_element "

go to previous element in list and return it

previous element in list or NULL if list is empty ";

%feature("docstring")  shogun::CList::get_current_element "

get current element in list

current element in list or NULL if not available ";

%feature("docstring")  shogun::CList::append_element "

append element AFTER the current element

Parameters:
-----------

data:  data element to append

if appending was successful ";

%feature("docstring")  shogun::CList::append_element_at_listend "

append at end of list

Parameters:
-----------

data:  data element to append

if appending was successful ";

%feature("docstring")  shogun::CList::insert_element "

insert element BEFORE the current element

Parameters:
-----------

data:  data element to insert

if inserting was successful ";

%feature("docstring")  shogun::CList::delete_element "

erases current element the new current element is the successor of the
former current element

the elements data - if available - is returned else NULL ";

%feature("docstring")  shogun::CList::get_name "

object name ";


// File: classshogun_1_1CListElement.xml
%feature("docstring") shogun::CListElement "

Class ListElement, defines how an element of the the list looks like.

C++ includes: List.h ";

%feature("docstring")  shogun::CListElement::CListElement "

constructor

Parameters:
-----------

p_data:  data of this element

p_prev:  previous element

p_next:  next element ";

%feature("docstring")  shogun::CListElement::~CListElement "

destructor ";


// File: classshogun_1_1CMath.xml
%feature("docstring") shogun::CMath "

Class which collects generic mathematical functions.

C++ includes: Mathematics.h ";

/*  constants  */

/*

*/

/*  Constructor/Destructor.  */

/*

*/

%feature("docstring")  shogun::CMath::CMath "

Constructor - initializes log-table. ";

%feature("docstring")  shogun::CMath::~CMath "

Destructor - frees logtable. ";

/*  min/max/abs functions.  */

/*

*/

%feature("docstring")  shogun::CMath::min "

return the minimum of two integers ";

%feature("docstring")  shogun::CMath::max "

return the maximum of two integers ";

%feature("docstring")  shogun::CMath::clamp "

return the value clamped to interval [lb,ub] ";

%feature("docstring")  shogun::CMath::abs "

return the maximum of two integers ";

/*  misc functions  */

/*

*/

%feature("docstring")  shogun::CMath::round "";

%feature("docstring")  shogun::CMath::floor "";

%feature("docstring")  shogun::CMath::ceil "";

%feature("docstring")  shogun::CMath::sign "

signum of type T variable a ";

%feature("docstring")  shogun::CMath::swap "

swap e.g. floats a and b ";

%feature("docstring")  shogun::CMath::resize "

resize array from old_size to new_size (keeping as much array content
as possible intact) ";

%feature("docstring")  shogun::CMath::twonorm "

|| x ||_2 ";

%feature("docstring")  shogun::CMath::qsq "

|| x ||_q^q ";

%feature("docstring")  shogun::CMath::qnorm "

|| x ||_q ";

%feature("docstring")  shogun::CMath::sq "

x^2 ";

%feature("docstring")  shogun::CMath::sqrt "

x^0.5 ";

%feature("docstring")  shogun::CMath::sqrt "

x^0.5 ";

%feature("docstring")  shogun::CMath::sqrt "

x^0.5 ";

%feature("docstring")  shogun::CMath::powl "

x^n ";

%feature("docstring")  shogun::CMath::pow "";

%feature("docstring")  shogun::CMath::pow "";

%feature("docstring")  shogun::CMath::pow "";

%feature("docstring")  shogun::CMath::exp "";

%feature("docstring")  shogun::CMath::log10 "";

%feature("docstring")  shogun::CMath::log2 "";

%feature("docstring")  shogun::CMath::log "";

%feature("docstring")  shogun::CMath::transpose_matrix "";

%feature("docstring")  shogun::CMath::pinv "

return the pseudo inverse for matrix when matrix has shape (rows,
cols) the pseudo inverse has (cols, rows) ";

%feature("docstring")  shogun::CMath::dgemm "";

%feature("docstring")  shogun::CMath::dgemv "";

%feature("docstring")  shogun::CMath::factorial "";

%feature("docstring")  shogun::CMath::init_random "";

%feature("docstring")  shogun::CMath::random "";

%feature("docstring")  shogun::CMath::random "";

%feature("docstring")  shogun::CMath::random "";

%feature("docstring")  shogun::CMath::random "";

%feature("docstring")  shogun::CMath::clone_vector "";

%feature("docstring")  shogun::CMath::fill_vector "";

%feature("docstring")  shogun::CMath::range_fill_vector "";

%feature("docstring")  shogun::CMath::random_vector "";

%feature("docstring")  shogun::CMath::randperm "";

%feature("docstring")  shogun::CMath::nchoosek "";

%feature("docstring")  shogun::CMath::vec1_plus_scalar_times_vec2 "

x=x+alpha*y ";

%feature("docstring")  shogun::CMath::dot "

compute dot product between v1 and v2 (blas optimized) ";

%feature("docstring")  shogun::CMath::dot "

compute dot product between v1 and v2 (blas optimized) ";

%feature("docstring")  shogun::CMath::dot "

compute dot product between v1 and v2 (blas optimized) ";

%feature("docstring")  shogun::CMath::dot "

compute dot product between v1 and v2 (blas optimized) ";

%feature("docstring")  shogun::CMath::dot "

compute dot product between v1 and v2 (for 64bit unsigned ints) ";

%feature("docstring")  shogun::CMath::dot "

compute dot product between v1 and v2 (for 64bit ints) ";

%feature("docstring")  shogun::CMath::dot "

compute dot product between v1 and v2 (for 32bit ints) ";

%feature("docstring")  shogun::CMath::dot "

compute dot product between v1 and v2 (for 32bit unsigned ints) ";

%feature("docstring")  shogun::CMath::dot "

compute dot product between v1 and v2 (for 16bit unsigned ints) ";

%feature("docstring")  shogun::CMath::dot "

compute dot product between v1 and v2 (for 16bit unsigned ints) ";

%feature("docstring")  shogun::CMath::dot "

compute dot product between v1 and v2 (for 8bit (un)signed ints) ";

%feature("docstring")  shogun::CMath::dot "

compute dot product between v1 and v2 (for 8bit (un)signed ints) ";

%feature("docstring")  shogun::CMath::dot "

compute dot product between v1 and v2 ";

%feature("docstring")  shogun::CMath::add "

target=alpha*vec1 + beta*vec2 ";

%feature("docstring")  shogun::CMath::add_scalar "

add scalar to vector inplace ";

%feature("docstring")  shogun::CMath::scale_vector "

scale vector inplace ";

%feature("docstring")  shogun::CMath::sum "

return sum(vec) ";

%feature("docstring")  shogun::CMath::max "

return max(vec) ";

%feature("docstring")  shogun::CMath::sum_abs "

return sum(abs(vec)) ";

%feature("docstring")  shogun::CMath::fequal "

return sum(abs(vec)) ";

%feature("docstring")  shogun::CMath::mean "";

%feature("docstring")  shogun::CMath::trace "";

%feature("docstring")  shogun::CMath::sort "

performs a bubblesort on a given matrix a. it is sorted in ascending
order from top to bottom and left to right ";

%feature("docstring")  shogun::CMath::sort "";

%feature("docstring")  shogun::CMath::radix_sort "

performs a in-place radix sort in ascending order ";

%feature("docstring")  shogun::CMath::byte "";

%feature("docstring")  shogun::CMath::radix_sort_helper "";

%feature("docstring")  shogun::CMath::insertion_sort "

performs insertion sort of an array output of length size it is sorted
from in ascending (for type T) ";

%feature("docstring")  shogun::CMath::qsort "

performs a quicksort on an array output of length size it is sorted
from in ascending (for type T) ";

%feature("docstring")  shogun::CMath::display_bits "

display bits (useful for debugging) ";

%feature("docstring")  shogun::CMath::display_vector "

display vector (useful for debugging) ";

%feature("docstring")  shogun::CMath::display_matrix "

display matrix (useful for debugging) ";

%feature("docstring")  shogun::CMath::qsort_index "

performs a quicksort on an array output of length size it is sorted in
ascending order (for type T1) and returns the index (type T2) matlab
alike [sorted,index]=sort(output) ";

%feature("docstring")  shogun::CMath::qsort_backward_index "

performs a quicksort on an array output of length size it is sorted in
ascending order (for type T1) and returns the index (type T2) matlab
alike [sorted,index]=sort(output) ";

%feature("docstring")  shogun::CMath::parallel_qsort_index "

performs a quicksort on an array output of length size it is sorted in
ascending order (for type T1) and returns the index (type T2) matlab
alike [sorted,index]=sort(output)

parallel version ";

%feature("docstring")  shogun::CMath::parallel_qsort_index "";

%feature("docstring")  shogun::CMath::min "";

%feature("docstring")  shogun::CMath::nmin "";

%feature("docstring")  shogun::CMath::unique "";

%feature("docstring")  shogun::CMath::binary_search_helper "";

%feature("docstring")  shogun::CMath::binary_search "";

%feature("docstring")  shogun::CMath::binary_search_max_lower_equal "";

%feature("docstring")  shogun::CMath::Align "

align two sequences seq1 & seq2 of length l1 and l2 using gapCost
return alignment cost ";

%feature("docstring")  shogun::CMath::calcroc "

calculates ROC into (fp,tp) from output and label of length size
returns index with smallest error=fp+fn ";

/*  summing functions  */

/*

*/

%feature("docstring")  shogun::CMath::logarithmic_sum "

sum logarithmic probabilities. Probability measures are summed up but
are now given in logspace where direct summation of exp(operand) is
not possible due to numerical problems, i.e. eg. exp(-1000)=0.
Therefore we do log( exp(a) + exp(b)) = a + log (1 + exp (b-a)) where
a = max(p,q) and b min(p,q). ";

%feature("docstring")  shogun::CMath::get_name "

object name ";


// File: classshogun_1_1CMemoryMappedFile.xml
%feature("docstring") shogun::CMemoryMappedFile "

memory mapped file

Implements a memory mapped file for super fast file access.

C++ includes: MemoryMappedFile.h ";

%feature("docstring")  shogun::CMemoryMappedFile::CMemoryMappedFile "

constructor

open a memory mapped file for read or read/write mode

Parameters:
-----------

fname:  name of file, zero terminated string

flag:  determines read or read write mode (can be 'r' or 'w')

fsize:  overestimate of expected file size (in bytes) when opened in
write mode; Underestimating the file size will result in an error to
occur upon writing. In case the exact file size is known later on, it
can be reduced via set_truncate_size() before closing the file. ";

%feature("docstring")  shogun::CMemoryMappedFile::~CMemoryMappedFile "

destructor ";

%feature("docstring")  shogun::CMemoryMappedFile::get_map "

get the mapping address It can now be accessed via, e.g.

double* x = get_map() x[index]= foo; (for write mode) foo = x[index];
(for read and write mode)

length of file ";

%feature("docstring")  shogun::CMemoryMappedFile::get_length "

get the number of objects of type T cointained in the file

length of file ";

%feature("docstring")  shogun::CMemoryMappedFile::get_size "

get the size of the file in bytes

size of file in bytes ";

%feature("docstring")  shogun::CMemoryMappedFile::get_line "

get next line from file

The returned line may be modfied in case the file was opened
read/write. It is otherwise read-only.

Parameters:
-----------

len:  length of line (returned via reference)

offs:  offset to be passed for reading next line, should be 0
initially (returned via reference)

line (NOT ZERO TERMINATED) ";

%feature("docstring")  shogun::CMemoryMappedFile::write_line "

write line to file

Parameters:
-----------

line:  string to be written (must not contain ' ' and not required to
be zero terminated)

len:  length of the string to be written

offs:  offset to be passed for writing next line, should be 0
initially (returned via reference)

line (NOT ZERO TERMINATED) ";

%feature("docstring")  shogun::CMemoryMappedFile::set_truncate_size "

set file size

When the file is opened for read/write mode, it will be truncated upon
destruction of the CMemoryMappedFile object. This is automagically
determined when writing lines, but might have to be set manually for
other data types, which is what this function is for.

Parameters:
-----------

sz:  byte number at which to truncate the file, zero to disable file
truncation. Has an effect only when file is opened with in read/write
mode 'w' ";

%feature("docstring")  shogun::CMemoryMappedFile::get_num_lines "

count the number of lines in a file

number of lines ";

%feature("docstring")  shogun::CMemoryMappedFile::get_name "

object name ";


// File: structshogun_1_1ConsensusEntry.xml
%feature("docstring") shogun::ConsensusEntry "

consensus entry

C++ includes: Trie.h ";


// File: classshogun_1_1CSet.xml
%feature("docstring") shogun::CSet "

Template Set class.

Lazy implementation of a set. Set grows and shrinks dynamically and
can be conveniently iterated through via the [] operator.

C++ includes: Set.h ";

%feature("docstring")  shogun::CSet::CSet "

Default constructor ";

%feature("docstring")  shogun::CSet::~CSet "

Default destructor ";

%feature("docstring")  shogun::CSet::add "

Add an element to the set

Parameters:
-----------

e:  elemet to be added ";

%feature("docstring")  shogun::CSet::remove "

Remove an element from the set

Parameters:
-----------

e:  elemet to be removed ";

%feature("docstring")  shogun::CSet::contains "

Remove an element from the set

Parameters:
-----------

e:  elemet to be removed ";

%feature("docstring")  shogun::CSet::get_num_elements "

get number of elements

number of elements ";

%feature("docstring")  shogun::CSet::get_element "

get set element at index

(does NOT do bounds checking)

Parameters:
-----------

index:  index

array element at index ";

%feature("docstring")  shogun::CSet::get_name "

object name ";


// File: classshogun_1_1CSignal.xml
%feature("docstring") shogun::CSignal "

Class Signal implements signal handling to e.g. allow ctrl+c to cancel
a long running process.

This is done in two ways:

A signal handler is attached to trap the SIGINT and SIGURG signal.
Pressing ctrl+c or sending the SIGINT (kill ...) signal to the shogun
process will make shogun print a message asking to immediately exit
the running method and to fall back to the command line.

When an URG signal is received or ctrl+c P is pressed shogun will
prematurely stop a method and continue execution. For example when an
SVM solver takes a long time without progressing much, one might still
be interested in the result and should thus send SIGURG or
interactively prematurely stop the method

C++ includes: Signal.h ";

%feature("docstring")  shogun::CSignal::CSignal "

default constructor ";

%feature("docstring")  shogun::CSignal::~CSignal "";

%feature("docstring")  shogun::CSignal::get_name "

object name ";


// File: classshogun_1_1CSimpleFile.xml
%feature("docstring") shogun::CSimpleFile "

Template class SimpleFile to read and write from files.

Currently only simple reading and writing of blocks is supported.

C++ includes: SimpleFile.h ";

%feature("docstring")  shogun::CSimpleFile::CSimpleFile "

constructor rw is either r for read and w for write

Parameters:
-----------

fname:  filename

f:  file descriptor ";

%feature("docstring")  shogun::CSimpleFile::~CSimpleFile "";

%feature("docstring")  shogun::CSimpleFile::load "

load

Parameters:
-----------

target:  load target

num:  number of read elements

loaded target or NULL if unsuccessful ";

%feature("docstring")  shogun::CSimpleFile::save "

save

Parameters:
-----------

target:  target to save to

num:  number of elements to write

if saving was successful ";

%feature("docstring")  shogun::CSimpleFile::get_buffered_line "

read a line (buffered; to be implemented)

Parameters:
-----------

line:  linebuffer to write to

len:  maximum length ";

%feature("docstring")  shogun::CSimpleFile::free_line_buffer "

free the line buffer ";

%feature("docstring")  shogun::CSimpleFile::set_line_buffer_size "

set the size of the line buffer

Parameters:
-----------

bufsize:  size of the line buffer ";

%feature("docstring")  shogun::CSimpleFile::is_ok "

check if status is ok

if status is ok ";

%feature("docstring")  shogun::CSimpleFile::get_name "

object name ";


// File: classshogun_1_1CTime.xml
%feature("docstring") shogun::CTime "

Class Time that implements a stopwatch based on either cpu time or
wall clock time.

C++ includes: Time.h ";

%feature("docstring")  shogun::CTime::CTime "

constructor

Parameters:
-----------

start:  if time measurement shall be started ";

%feature("docstring")  shogun::CTime::~CTime "";

%feature("docstring")  shogun::CTime::cur_runtime "

get current cpu runtime

Parameters:
-----------

verbose:  if time shall be printed

current cpu runtime ";

%feature("docstring")  shogun::CTime::cur_runtime_diff "

get time difference between start and NOW

Parameters:
-----------

verbose:  if time difference shall be printed

time difference between start and NOW ";

%feature("docstring")  shogun::CTime::cur_runtime_diff_sec "

get time difference between start and NOW in seconds

Parameters:
-----------

verbose:  if time difference shall be printed

time difference between start and NOW in seconds ";

%feature("docstring")  shogun::CTime::start "

start the counter

Parameters:
-----------

verbose:  if start time shall be printed

start time in seconds ";

%feature("docstring")  shogun::CTime::cur_time_diff "

get time difference between start and NOW in seconds

Parameters:
-----------

verbose:  if time difference shall be printed

time difference between start and NOW in seconds ";

%feature("docstring")  shogun::CTime::time_diff_sec "

get time difference between start and stop in seconds

Parameters:
-----------

verbose:  if time difference shall be printed

time difference between start and stop in seconds ";

%feature("docstring")  shogun::CTime::stop "

stop the counter

Parameters:
-----------

verbose:  if stop time shall be printed

stop time in seconds ";

%feature("docstring")  shogun::CTime::get_name "

object name ";


// File: classshogun_1_1CTrie.xml
%feature("docstring") shogun::CTrie "

Template class Trie implements a suffix trie, i.e. a tree in which all
suffixes up to a certain length are stored.

It is excessively used in the CWeightedDegreeStringKernel and
CWeightedDegreePositionStringKernel to construct the whole features
space $\\\\Phi(x)$ and enormously helps here to speed up SVM training
and evaluation.

Note that depending on the underlying structure used, a single symbol
in the tree requires 20 bytes ( DNATrie). It is also used to do the
efficient recursion in computing positional oligomer importance
matrices (POIMs) where the structure requires * 20+3*8 ( POIMTrie)
bytes.

Finally note that this try may use compact internal nodes (for strings
that appear without modifications, thus not requiring further
branches), which may save a lot of memory on higher degree tries.

C++ includes: Trie.h ";

%feature("docstring")  shogun::CTrie::CTrie "

constructor

Parameters:
-----------

d:  degree

p_use_compact_terminal_nodes:  if compact terminal nodes shall be used
";

%feature("docstring")  shogun::CTrie::CTrie "

copy constructor ";

%feature("docstring")  shogun::CTrie::~CTrie "";

%feature("docstring")  shogun::CTrie::compare_traverse "

compare traverse

Parameters:
-----------

node:  node

other:  other trie

other_node:  other node

if comparison was successful ";

%feature("docstring")  shogun::CTrie::compare "

compare

Parameters:
-----------

other:  other trie

if comparison was successful ";

%feature("docstring")  shogun::CTrie::find_node "

find node

Parameters:
-----------

node:  node to find

trace:  trace

trace_len:  length of trace ";

%feature("docstring")  shogun::CTrie::find_deepest_node "

find deepest node

Parameters:
-----------

start_node:  start node

deepest_node:  deepest node will be stored in here

depth of deepest node ";

%feature("docstring")  shogun::CTrie::display_node "

display node

Parameters:
-----------

node:  node to display ";

%feature("docstring")  shogun::CTrie::destroy "

destroy ";

%feature("docstring")  shogun::CTrie::set_degree "

set degree

Parameters:
-----------

d:  new degree ";

%feature("docstring")  shogun::CTrie::create "

create

Parameters:
-----------

len:  length of new trie

p_use_compact_terminal_nodes:  if compact terminal nodes shall be used
";

%feature("docstring")  shogun::CTrie::delete_trees "

delete trees

Parameters:
-----------

p_use_compact_terminal_nodes:  if compact terminal nodes shall be used
";

%feature("docstring")  shogun::CTrie::add_to_trie "

add to trie

Parameters:
-----------

i:  i

seq_offset:  sequence offset

vec:  vector

alpha:  alpha

weights:  weights

degree_times_position_weights:  if degree times position weights shall
be applied ";

%feature("docstring")  shogun::CTrie::compute_abs_weights_tree "

compute absolute weights tree

Parameters:
-----------

tree:  tree to compute for

depth:  depth

computed absolute weights tree ";

%feature("docstring")  shogun::CTrie::compute_abs_weights "

compute absolute weights

Parameters:
-----------

len:  length

computed absolute weights ";

%feature("docstring")  shogun::CTrie::compute_by_tree_helper "

compute by tree helper

Parameters:
-----------

vec:  vector

len:  length

seq_pos:  sequence position

tree_pos:  tree position

weight_pos:  weight position

weights:

degree_times_position_weights:  if degree times position weights shall
be applied

a computed value ";

%feature("docstring")  shogun::CTrie::compute_by_tree_helper "

compute by tree helper

Parameters:
-----------

vec:  vector

len:  length

seq_pos:  sequence position

tree_pos:  tree position

weight_pos:  weight position

LevelContrib:  level contribution

factor:  factor

mkl_stepsize:  MKL stepsize

weights:

degree_times_position_weights:  if degree times position weights shall
be applied ";

%feature("docstring")  shogun::CTrie::compute_scoring_helper "

compute scoring helper

Parameters:
-----------

tree:  tree

i:  i

j:  j

weight:  weight

d:  degree

max_degree:  maximum degree

num_feat:  number of features

num_sym:  number of symbols

sym_offset:  symbol offset

offs:  offsets

result:  result ";

%feature("docstring")
shogun::CTrie::add_example_to_tree_mismatch_recursion "

add example to tree mismatch recursion

Parameters:
-----------

tree:  tree

i:  i

alpha:  alpha

vec:  vector

len_rem:  length of rem

degree_rec:  degree rec

mismatch_rec:  mismatch rec

max_mismatch:  maximum mismatch

weights:  weights ";

%feature("docstring")  shogun::CTrie::traverse "

traverse

Parameters:
-----------

tree:  tree

p:  p

info:  tree parse info

depth:  depth

x:  x

k:  k ";

%feature("docstring")  shogun::CTrie::count "

count

Parameters:
-----------

w:  w

depth:  depth

info:  tree parse info

p:  p

x:  x

k:  ";

%feature("docstring")  shogun::CTrie::compact_nodes "

compact nodes

Parameters:
-----------

start_node:  start node

depth:  depth

weights:  weights ";

%feature("docstring")  shogun::CTrie::get_cumulative_score "

get cumulative score

Parameters:
-----------

pos:  position

seq:  sequence

deg:  degree

weights:  weights

cumulative score ";

%feature("docstring")
shogun::CTrie::fill_backtracking_table_recursion "

fill backtracking table recursion

Parameters:
-----------

tree:  tree

depth:  depth

seq:  sequence

value:  value

table:  table of concensus entries

weights:  weights ";

%feature("docstring")  shogun::CTrie::fill_backtracking_table "

fill backtracking table

Parameters:
-----------

pos:  position

prev:  previous concencus entry

cur:  current concensus entry

cumulative:  if is cumulative

weights:  weights ";

%feature("docstring")  shogun::CTrie::POIMs_extract_W "

POIMs extract W

Parameters:
-----------

W:  W

K:  K ";

%feature("docstring")  shogun::CTrie::POIMs_precalc_SLR "

POIMs precalc SLR

Parameters:
-----------

distrib:  distribution ";

%feature("docstring")  shogun::CTrie::POIMs_get_SLR "

POIMs get SLR

Parameters:
-----------

parentIdx:  parent index

sym:  symbol

depth:  depth

S:  will point to S

L:  will point to L

R:  will point to R ";

%feature("docstring")  shogun::CTrie::POIMs_add_SLR "

POIMs add SLR

Parameters:
-----------

poims:  POIMs

K:  K

debug:  debug level ";

%feature("docstring")  shogun::CTrie::get_use_compact_terminal_nodes "

get use compact terminal nodes

if compact terminal nodes are used ";

%feature("docstring")  shogun::CTrie::set_use_compact_terminal_nodes "

set use compact terminal nodes

Parameters:
-----------

p_use_compact_terminal_nodes:  if compact terminal nodes shall be used
";

%feature("docstring")  shogun::CTrie::get_num_used_nodes "

get number of used nodes

number of used nodes ";

%feature("docstring")  shogun::CTrie::set_position_weights "

set position weights

Parameters:
-----------

p_position_weights:  new position weights ";

%feature("docstring")  shogun::CTrie::get_node "

get node

node ";

%feature("docstring")  shogun::CTrie::check_treemem "

check tree memory usage ";

%feature("docstring")  shogun::CTrie::set_weights_in_tree "

set weights in tree

Parameters:
-----------

weights_in_tree_:  if weights shall be in tree ";

%feature("docstring")  shogun::CTrie::get_weights_in_tree "

get weights in tree

if weights are in tree ";

%feature("docstring")  shogun::CTrie::POIMs_extract_W_helper "

POIMs extract W helper

Parameters:
-----------

nodeIdx:  node index

depth:  depth

offset:  offset

y0:  y0

W:  W

K:  K ";

%feature("docstring")  shogun::CTrie::POIMs_calc_SLR_helper1 "

POIMs calc SLR helper

Parameters:
-----------

distrib:  distribution

i:  i

nodeIdx:  node index

left_tries_idx:  left tries index

depth:  depth

lastSym:  last symbol

S:  S

L:  L

R:  R ";

%feature("docstring")  shogun::CTrie::POIMs_calc_SLR_helper2 "

POIMs calc SLR helper 2

Parameters:
-----------

distrib:  distribution

i:  i

nodeIdx:  node index

left_tries_idx:  left tries index

depth:  depth

S:  S

L:  L

R:  R ";

%feature("docstring")  shogun::CTrie::POIMs_add_SLR_helper1 "

POIMs add SLR helper 1

Parameters:
-----------

nodeIdx:  node index

depth:  depth

i:  i

y0:  y0

poims:  POIMs

K:  K

debug:  debug level ";

%feature("docstring")  shogun::CTrie::POIMs_add_SLR_helper2 "

POIMs add SLR helper 2

Parameters:
-----------

poims:  POIMs

K:  K

k:  k

i:  i

y:  y

valW:  value W

valS:  value S

valL:  value L

valR:  value R

debug:  debug level ";

%feature("docstring")  shogun::CTrie::get_name "

object name ";


// File: structshogun_1_1DNATrie.xml
%feature("docstring") shogun::DNATrie "

DNA trie

C++ includes: Trie.h ";


// File: structshogun_1_1CHash_1_1MD5Context.xml


// File: structshogun_1_1POIMTrie.xml
%feature("docstring") shogun::POIMTrie "

POIM trie

C++ includes: Trie.h ";


// File: structradix__stack__t.xml
%feature("docstring") radix_stack_t "

Stack structure

C++ includes: Mathematics.h ";


// File: classshogun_1_1ShogunException.xml
%feature("docstring") shogun::ShogunException "

Class ShogunException defines an exception which is thrown whenever an
error inside of shogun occurs.

C++ includes: ShogunException.h ";

%feature("docstring")  shogun::ShogunException::ShogunException "

constructor

Parameters:
-----------

str:  exception string ";

%feature("docstring")  shogun::ShogunException::~ShogunException "

destructor ";

%feature("docstring")  shogun::ShogunException::get_exception_string "

get exception string

the exception string ";


// File: structshogun_1_1CCache_1_1TEntry.xml


// File: structthread__qsort.xml
%feature("docstring") thread_qsort "

pair thread qsort

C++ includes: Mathematics.h ";


// File: structshogun_1_1TreeParseInfo.xml
%feature("docstring") shogun::TreeParseInfo "

tree parse info

C++ includes: Trie.h ";


// File: namespaceshogun.xml
%feature("docstring")  shogun::wrap_dsyev "";

%feature("docstring")  shogun::wrap_dgesvd "";


// File: Array_8h.xml


// File: Array2_8h.xml


// File: Array3_8h.xml


// File: BinaryStream_8h.xml


// File: BitString_8h.xml


// File: Cache_8h.xml


// File: common_8h.xml
/*  Standard Types  */

/*  Definition of Platform independent Types

*/


// File: Compressor_8h.xml


// File: config_8h.xml


// File: Cplex_8h.xml


// File: DynamicArray_8h.xml


// File: DynInt_8h.xml


// File: File_8h.xml


// File: GCArray_8h.xml


// File: Hash_8h.xml


// File: IndirectObject_8h.xml


// File: io_8h.xml


// File: lapack_8h.xml
%feature("docstring")  shogun::dsyev_ "";

%feature("docstring")  shogun::dgesvd_ "";

%feature("docstring")  shogun::dposv_ "";

%feature("docstring")  shogun::dpotrf_ "";


// File: List_8h.xml


// File: Mathematics_8h.xml


// File: memory_8h.xml


// File: MemoryMappedFile_8h.xml


// File: Set_8h.xml


// File: ShogunException_8h.xml


// File: Signal_8h.xml


// File: SimpleFile_8h.xml


// File: Time_8h.xml


// File: Trie_8h.xml


// File: versionstring_8h.xml


// File: dir_1c98848a2231562b6924fbfa41d69ceb.xml


// File: dir_560ddd44a5cfd1053a96f217f73c4ff4.xml

