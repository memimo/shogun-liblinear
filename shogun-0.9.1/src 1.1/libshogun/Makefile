# note that the word libshogun (all lowercase or all uppercase)
# will be replaced with the current interface name
#
.SUFFIXES: .c .c.o .cpp .cpp.o .h .i .so .dylib .a _wrap.cxx _wrap.cxx.o
.PHONY: all out clean end targets warn2err err2warn ctags buildout buildend dist

SRCDIR = .

ifneq (../.config,$(wildcard ../.config))
all:
	@echo !!! Please run ./configure in the top level directory with the appropriate options first !!!
else
include ../.config
ifeq (../.config-local,$(wildcard ../.config-local))
include ../.config-local
endif

ifeq (libshogun,$(findstring libshogun,$(LIBSHOGUN_TARGET)))
	LIBNAME=$(shell echo $(LIBSHOGUN_TARGET) | cut -f 1 -d '.')
	SONAME=$(LIBSHOGUN_SONAME)
	OTHER_TARGETS=$(LIBNAME).a
	LIBSYMLINKCMD=rm -f $(LIBNAME).$(EXT_LIB) $(SONAME); \
				  ln -s $(LIBSHOGUN_TARGET) $(SONAME) ; \
				  ln -s $(LIBSHOGUN_TARGET) $(LIBNAME).$(EXT_LIB)

ifeq (libshogun.,$(findstring libshogun.,$(LIBSHOGUN_TARGET)))
DEPDEPS=lib/versionstring.h
endif

else

ifeq (elwms,$(findstring elwms,$(LIBSHOGUN_TARGET)))
	LIBSYMLINKCMD=rm -f elwms.oct elwms.mex*; ln -s $(LIBSHOGUN_TARGET) elwms.oct; \
				  if test -n "$(MATLAB_TARGET)";\
				  then \
				    ln -s $(LIBSHOGUN_TARGET) \
				      elwms.$(shell echo $(MATLAB_TARGET) | cut -f 2 -d '.');\
				  fi
endif

ifeq (modular,$(findstring modular,$(LIBSHOGUN_TARGET)))
INCLUDES += $(INCLUDES_MODULAR_INTERFACES)
LINKFLAGS += $(LINKFLAGS_MODULAR_INTERFACES)
SWIGFLAGS += $(SWIGFLAGS_LIBSHOGUN) $(INCLUDES_MODULAR_INTERFACES) -I.. -I.
OTHERDEPS = $(SWIGSRCFILES) $(SWIGLIBFILES) 
ifeq (yes,$(USE_DOXYGEN))
OTHERDEPS += $(SWIGDOCFILES)
endif
$(shell for i in ../modular/*.* ; do test -f `basename $$i` || ln -s $$i . ; done )
else
	INCLUDES += $(INCLUDES_STATIC_INTERFACES)
	LINKFLAGS += $(LINKFLAGS_STATIC_INTERFACES)
endif

ifeq (r_modular,$(LIBSHOGUN_TARGET))
OTHERDEPS += $(SWIGRFILES) $(SWIGRDATAFILES)
endif

endif

all: $(LIBSHOGUN_TARGET) $(OTHER_TARGETS) $(OTHERDEPS)
endif

INCLUDES += $(INCLUDES_LIBSHOGUN)
DEFINES += $(DEFINES_LIBSHOGUN)
COMPFLAGS_CPP += $(DEFINES)
COMPFLAGS_C += $(DEFINES)
LINKFLAGS += $(LINKFLAGS_LIBSHOGUN)
POSTLINKFLAGS += $(POSTLINKFLAGS_LIBSHOGUN)

# SRC/OBJFILES lie all in the subdir of SRCDIR
SVNEntries = $(shell find $(SRCDIR) -name entries -type f)
HEADERFILES	= $(shell find $(SRCDIR) -name "*.$(EXT_SRC_HEADER)" )
SRCFILES 	= $(shell find $(SRCDIR) -name "*.$(EXT_SRC_C)" -o -name "*.$(EXT_SRC_CPP)")

ifeq (modular,$(findstring modular,$(LIBSHOGUN_TARGET)))
OBJFILES 	= $(patsubst %.$(EXT_SRC_CPP),%.$(EXT_OBJ_CPP), $(shell find $(SRCDIR) -name "*.$(EXT_SRC_CPP)" )) \
			  $(patsubst %.$(EXT_SRC_C),%.$(EXT_OBJ_C), $(shell find $(SRCDIR) -name "*.$(EXT_SRC_C)" )) \
			  $(patsubst %.$(EXT_INP_SWIG),%$(EXT_OBJ_SWIG), $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \;))
else
OBJFILES 	= $(patsubst %.$(EXT_SRC_CPP),%.$(EXT_OBJ_CPP), $(shell find $(SRCDIR) -name "*.$(EXT_SRC_CPP)")) \
			  $(patsubst %.$(EXT_SRC_C),%.$(EXT_OBJ_C), $(shell find $(SRCDIR) -name "*.$(EXT_SRC_C)"))
endif

SWIGIFILES   	= $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)")
SWIGSRCFILES 	= $(patsubst %.$(EXT_INP_SWIG),%$(EXT_SRC_SWIG), \
				  $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \;))
SWIGDOCFILES 	= $(patsubst %.$(EXT_INP_SWIG),%$(EXT_SWIG_DOXY), \
				  $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \;))
SWIGLIBFILES 	= $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \; | \
				  sed 's=/\([^/]*\.\)$(EXT_INP_SWIG)=/$(PRE_LIB_SWIG_LIBSHOGUN)\1$(EXT_LIB_SWIG_LIBSHOGUN)=' | sort)
SWIGPYFILES 	= $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \; | \
				  sed 's=/\([^/]*\.\)$(EXT_INP_SWIG)=/\1py=')
SWIGRFILES 	= $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \; | \
			  sed 's=/\([^/]*\.\)$(EXT_INP_SWIG)=/\1R=')
SWIGRDATAFILES 	= $(shell find $(SRCDIR) -name "*.$(EXT_INP_SWIG)" -exec grep -le '^%module' {} \; | \
				  sed 's=/\([^/]*\.\)$(EXT_INP_SWIG)=/\1RData=')
TAR:=../shogun-$(shell date +%Y-%m-%d).tar

PRELPATH=$(shell python ../.relpath.py $(DESTDIR)$(PYDIR) $(DESTDIR)$(LIBDIR)/shogun)
ORELPATH=$(shell python ../.relpath.py $(DESTDIR)$(OCTDIR) $(DESTDIR)$(LIBDIR)/shogun)
RRELPATH=$(shell python ../.relpath.py $(DESTDIR)$(RDIR)/elwms/libs $(DESTDIR)$(LIBDIR)/shogun)

warn2err: $(SRCFILES)
	for i in $^ ; do cat $$i | sed 's/^\#warning/#error/g' >$$i.new ; mv $$i.new $$i ; done
err2warn: $(SRCFILES)
	for i in $^ ; do cat $$i | sed 's/^\#error/#warning/g' >$$i.new ; mv $$i.new $$i ; done

ctags: tags

tags:	$(SRCFILES)
	ctags */*.$(EXT_SRC_CPP) */*.h */*.$(EXT_SRC_C)

linecount:
	@echo -n "C/C++ source code files (.cpp,.c):"
	@cat $(SRCFILES) | grep -vc '^ *$$'
	@echo -n "C/C++ header files (.h):"
	@cat $(HEADERFILES) | grep -vc '^ *$$'
	@echo -n "SWIG header files (.i):"
	@cat $(SWIGIFILES) | grep -vc '^ *$$'
	@echo -n "Total:"
	@cat $(SRCFILES) $(HEADERFILES) $(SWIGIFILES) | grep -vc '^ *$$'

targets:
	@echo "=========="
	@echo "=targets:="
	@echo "=========="
	@egrep '^[a-z]+:' Makefile | grep -v '\.o:' | \
		sed -e 's/:.*//g' -e 's/out//g' -e 's/end//g' -e 's/^/     /g' | sort | uniq

distclean: clean
		@rm -f tags .depend

swigclean:
	@rm -f $(shell find -name "*_wrap.h" -o -name "*_wrap.cxx.o" -o -name "*_wrap.cxx" 2>/dev/null) 

swigdocclean:
	@rm -rf $(shell find -name doxygen_xml -o -name "*_doxygen.i" 2>/dev/null) 

clean: swigclean swigdocclean
	@rm -f $(shell find . -name "*.cpp.o" -o -name "*.c.o")
	@rm -f $(shell find . -name "*.a" -o -name "*.so*" -o -name "*.dylib" -o -name "*.dll" -o -name "*~" -o -name ".#*")
	@rm -f $(shell find . -type f -name '*.py' ! -name '__*.py' ! -name '.*.py' | grep -v svn)
	@rm -f $(shell find . -type f -name '*.pyc' -o -name '*.R')
	@rm -f $(shell find . -type f -name '*.RData')
	@rm -f $(shell find . -type f -name '*.oct')
	@rm -f $(shell find . -type l -name '*.i')
	@rm -f $(shell for i in `find . -type l -name '*.doxy'` ; do rm -rf `basename $$i .doxy` ; done)
	@rm -f $(shell find . -type l -name '*.doxy')
	@rm -f $(shell find . -type f -name '*doxygen.i')
	@rm -f $(shell find . -type f -name '*wrap.cxx*')
	@rm -f $(shell find . -type f -name '*.swp')
	@rm -f $(shell find . -type f -name 'octave-core')
	@rm -f sg.mex* shogun shogun.exe gmon.out .depend configure.log .cpuinfo ./configure-* cplex.log
	@rm -f elwms.mex* elwms.so elwms.dylib elwms.dll elwms.oct
	@rm -f $(LIBSHOGUN_TARGET)

install: install-libshogun

install-cmdline: $(LIBSHOGUN_TARGET)
	install -d -m755 $(DESTDIR)$(BINDIR)
	install -m755 $(LIBSHOGUN_TARGET) $(DESTDIR)$(BINDIR)

install-libshogunui: $(LIBSHOGUN_TARGET)
	install -d -m755 $(DESTDIR)$(LIBDIR)
	install -d -m755 "$(DESTDIR)$(INCDIR)/shogun/ui"
	install -m755 $(LIBSHOGUN_TARGET) $(DESTDIR)$(LIBDIR)
	install -m755 libshogunui.a $(DESTDIR)$(LIBDIR)
	install -m644 *.h "$(DESTDIR)$(INCDIR)/shogun/ui"
	find "$(DESTDIR)$(INCDIR)/shogun/ui" -name '*.h' -exec $(SEDMI) \
		's/#include "\([^"]*\)"/#include <shogun\/ui\/\1>/' {} \;
	rm -f $(DESTDIR)$(LIBDIR)/libshogunui.$(EXT_LIB) $(DESTDIR)$(LIBDIR)/$(LIBSHOGUN_SONAME)
	cd $(DESTDIR)$(LIBDIR) && ( ln -s $(LIBSHOGUN_TARGET) libshogunui.$(EXT_LIB); \
	ln -s $(LIBSHOGUN_TARGET) $(LIBSHOGUN_SONAME) )
	-ldconfig

install-libshogun: $(LIBSHOGUN_TARGET)
	install -d -m755 $(DESTDIR)$(LIBDIR)
	install -d -m755 "$(DESTDIR)$(INCDIR)/shogun"
	install -m755 $(LIBSHOGUN_TARGET) $(DESTDIR)$(LIBDIR)
	install -m755 libshogun.a $(DESTDIR)$(LIBDIR)

	find ./ -mindepth 1 -type d ! \( -path '*.svn*' -o \
		-name '*interface*' -o -name '*guilib*' -o -name '*doxygen*' \)\
		-exec install -d -m755 "$(DESTDIR)$(INCDIR)/shogun/"\{\} \;
	find ./ -mindepth 1 -type d ! \( -path '*.svn*' -o -name '*wrap*' -o \
		-name '*interface*' -o -name '*guilib*' -o -name '*doxygen*' \) \
		-exec sh -c "install -m644 {}/*.h $(DESTDIR)$(INCDIR)/shogun/{}" \;
	find "$(DESTDIR)$(INCDIR)/shogun/" -name '*.h' \
		-exec $(SEDMI) 's/#include "\([^"]*\)"/#include <shogun\/\1>/' {} \;
	rm -f $(DESTDIR)$(LIBDIR)/libshogun.$(EXT_LIB) $(DESTDIR)$(LIBDIR)/$(LIBSHOGUN_SONAME)
	cd $(DESTDIR)$(LIBDIR) && ( ln -s $(LIBSHOGUN_TARGET) libshogun.$(EXT_LIB); \
	ln -s $(LIBSHOGUN_TARGET) $(LIBSHOGUN_SONAME) )
	-ldconfig

install-python: $(LIBSHOGUN_TARGET)
	install -d -m755 $(DESTDIR)$(PYDIR)
	install -m755 $(LIBSHOGUN_TARGET) $(DESTDIR)$(PYDIR)

install-octave: $(LIBSHOGUN_TARGET)
	install -d -m755 $(DESTDIR)$(OCTDIR)
	install -m755 $(LIBSHOGUN_TARGET) $(DESTDIR)$(OCTDIR)

install-r: $(LIBSHOGUN_TARGET)
	install -d -m755 $(DESTDIR)$(RDIR)/sg/demo
	install -d -m755 $(DESTDIR)$(RDIR)/sg/libs
	install -d -m755 $(DESTDIR)$(RDIR)/sg/Meta
	install -d -m755 $(DESTDIR)$(RDIR)/sg/R
	install -m755 $(LIBSHOGUN_TARGET) $(DESTDIR)$(RDIR)/sg/libs
	install -m644 ../../examples/r/*.R ../../examples/r/graphical/*.R $(DESTDIR)$(RDIR)/sg/demo
	../.r-install.sh $(DESTDIR)$(RDIR) sg $(EXT_LIB)

install-elwms: $(LIBSHOGUN_TARGET)
	#one .so file
ifneq ($(CMDLINE_TARGET),)
	install -d -m755 $(DESTDIR)$(LIBDIR)/shogun
	install -m755 $(LIBSHOGUN_TARGET) $(DESTDIR)$(LIBDIR)/shogun
endif
	#python
ifneq ($(PYTHON_TARGET),)
	rm -f $(DESTDIR)$(PYDIR)/$(LIBSHOGUN_TARGET) 
	install -d -m755 $(DESTDIR)$(PYDIR)
	cd $(DESTDIR)$(PYDIR) && ln -s $(PRELPATH)/$(LIBSHOGUN_TARGET)
endif
	#octave
ifneq ($(OCTAVE_TARGET),)
	install -d -m755 $(DESTDIR)$(OCTDIR)
	rm -f $(DESTDIR)$(OCTDIR)/elwms.oct
	cd $(DESTDIR)$(OCTDIR) && ln -s $(ORELPATH)/$(LIBSHOGUN_TARGET) elwms.oct
endif
	#r
ifneq ($(R_TARGET),)
	install -d -m755 $(DESTDIR)$(RDIR)/elwms/demo
	install -d -m755 $(DESTDIR)$(RDIR)/elwms/libs
	install -d -m755 $(DESTDIR)$(RDIR)/elwms/Meta
	install -d -m755 $(DESTDIR)$(RDIR)/elwms/R
	rm -f $(DESTDIR)$(RDIR)/elwms/libs/$(LIBSHOGUN_TARGET)
	cd $(DESTDIR)$(RDIR)/elwms/libs && ln -s $(RRELPATH)/$(LIBSHOGUN_TARGET) 
	install -m644 ../../examples/r/*.R ../../examples/r/graphical/*.R \
		$(DESTDIR)$(RDIR)/elwms/demo
	../.r-install.sh $(DESTDIR)$(RDIR) elwms $(EXT_LIB)
endif

install-matlab: $(LIBSHOGUN_TARGET)
	@echo "The matlab interface needs to be installed"
	@echo "using a different mechanism:"
	@echo
	@echo " just copy the sg.mex* to a <dir> of your choice"
	@echo " and addpath <dir>"
	@echo

install-python_modular: $(LIBSHOGUN_TARGET)
	install -d -m755 "$(DESTDIR)$(PYDIR)/shogun"
	install -m644 $(SWIGLIBFILES) $(SWIGPYFILES) "$(DESTDIR)$(PYDIR)/shogun"
	echo '__all__= [' > "$(DESTDIR)$(PYDIR)/shogun/__init__.py"
	for i in $(SWIGPYFILES) ; do echo $$i \
		| sed -e 's=.*/="=' -e 's/\.py/", /' \
		>>$(DESTDIR)$(PYDIR)/shogun/__init__.py; done
	echo ']' >> "$(DESTDIR)$(PYDIR)/shogun/__init__.py"

install-r_modular: $(LIBSHOGUN_TARGET)
	install -d -m755 $(DESTDIR)$(RDIR)/shogun/demo
	install -d -m755 $(DESTDIR)$(RDIR)/shogun/libs
	install -d -m755 $(DESTDIR)$(RDIR)/shogun/Meta
	install -d -m755 $(DESTDIR)$(RDIR)/shogun/R
	install -m644 ../../examples/r_modular/*.R $(DESTDIR)$(RDIR)/shogun/demo
	install -m755 *.$(EXT_LIB) $(DESTDIR)$(RDIR)/shogun/libs
	install -m755 *.RData $(DESTDIR)$(RDIR)/shogun/R
	../.r-install.sh $(DESTDIR)$(RDIR) shogun $(EXT_LIB)

install-octave_modular: $(LIBSHOGUN_TARGET)
	install -d -m755 "$(DESTDIR)$(OCTDIR)/shogun"
	install -m755 $(SWIGLIBFILES) "$(DESTDIR)$(OCTDIR)/shogun"
	(umask 0022 ; echo '%Initialize shogun by loading all of its modules' \
		> "$(DESTDIR)$(OCTDIR)/shogun/init_shogun.m" )
	for i in $(SWIGLIBFILES) ; do echo $$i | sed -e 's=.*/==' -e 's/\.oct/;/' \
		>>$(DESTDIR)$(OCTDIR)/shogun/init_shogun.m; done

run-testsuite: run-testsuite-libshogun
check-examples: check-examples-libshogun

run-testsuite-libshogun:
	true
check-examples-libshogun:
	cd ../../examples/libshogun && $(MAKE) clean && \
		$(MAKE) LIBRARY_PATH="$(LIBRARY_PATH)" LIB_PATH="$(DESTDIR)$(LIBDIR)" \
		INC_PATH="$(DESTDIR)$(INCDIR)" LIBS="$(LINKFLAGS) -lshogun" check-examples

run-testsuite-libshogunui:
	true
check-examples-libshogunui:
	true

run-testsuite-elwms:
	true
check-examples-elwms:
	true

run-testsuite-python:
	cd ../../testsuite/python && \
		( $(LIBRARY_PATH)=$(DESTDIR)$(LIBDIR) PYTHON=$(PYTHON) \
		PYTHONPATH="$(DESTDIR)$(PYDIR)" ./test_all.sh	)
check-examples-python:
	cd ../../examples/python && \
		( $(LIBRARY_PATH)=$(DESTDIR)$(LIBDIR) PYTHON=$(PYTHON) \
		PYTHONPATH="$(DESTDIR)$(PYDIR)" ./check.sh	)

run-testsuite-r:
	cd ../../testsuite/r && \
		( $(LIBRARY_PATH)=$(DESTDIR)$(LIBDIR) \
		R_LIBS="$(DESTDIR)$(RDIR)" ./test_all.sh )
check-examples-r:
	cd ../../examples/r && \
		( $(LIBRARY_PATH)=$(DESTDIR)$(LIBDIR) \
		R_LIBS="$(DESTDIR)$(RDIR)" ./check.sh )

run-testsuite-cmdline:
	cd ../../testsuite/cmdline && \
		( $(LIBRARY_PATH)=$(DESTDIR)$(LIBDIR) \
		PATH="$(DESTDIR)$(BINDIR)"  ./test_all.sh )
check-examples-cmdline:
	cd ../../examples/cmdline && \
		( $(LIBRARY_PATH)=$(DESTDIR)$(LIBDIR) \
		PATH="$(DESTDIR)$(BINDIR)"  ./check.sh )

run-testsuite-octave:
	cd ../../testsuite/octave && \
		( $(LIBRARY_PATH)=$(DESTDIR)$(LIBDIR) \
		OCTAVE_LOADPATH="$(DESTDIR)$(OCTDIR):" ./test_all.sh - octave )
check-examples-octave:
	cd ../../examples/octave && \
		( $(LIBRARY_PATH)=$(DESTDIR)$(LIBDIR) \
		OCTAVE_LOADPATH="$(DESTDIR)$(OCTDIR):" ./check.sh )

run-testsuite-matlab:
	cd ../../testsuite/matlab && \
		( ln -sf ../../src/matlab/$(LIBSHOGUN_TARGET) ; \
		$(LIBRARY_PATH)=$(DESTDIR)$(LIBDIR) ./test_all.sh - matlab)
check-examples-matlab:
	cd ../../examples/matlab && ./check.sh

run-testsuite-python_modular:
	cd ../../testsuite/python_modular && \
		( $(LIBRARY_PATH)=$(DESTDIR)$(LIBDIR) PYTHON=$(PYTHON) \
		PYTHONPATH="$(DESTDIR)$(PYDIR)" ./test_all.sh    )
check-examples-python_modular:
	cd ../../examples/python_modular && \
		( $(LIBRARY_PATH)=$(DESTDIR)$(LIBDIR) PYTHON=$(PYTHON) \
		PYTHONPATH="$(DESTDIR)$(PYDIR)" ./check.sh    )

run-testsuite-r_modular:
	cd ../../testsuite/r_modular && \
		( $(LIBRARY_PATH)=$(DESTDIR)$(LIBDIR) \
		R_LIBS="$(DESTDIR)$(RDIR)" ./test_all.sh )
check-examples-r_modular:
	cd ../../examples/r_modular && \
		( $(LIBRARY_PATH)=$(DESTDIR)$(LIBDIR) \
		R_LIBS="$(DESTDIR)$(RDIR)" ./check.sh )

run-testsuite-octave_modular:
	cd ../../testsuite/octave_modular && \
		( $(LIBRARY_PATH)=$(DESTDIR)$(LIBDIR) \
		OCTAVE_LOADPATH="$(DESTDIR)$(OCTDIR)/shogun:" ./test_all.sh )
check-examples-octave_modular:
	cd ../../examples/octave_modular && \
		( $(LIBRARY_PATH)=$(DESTDIR)$(LIBDIR) \
		OCTAVE_LOADPATH="$(DESTDIR)$(OCTDIR)/shogun:" ./check.sh )

lib/versionstring.h:    $(SVNEntries) 
	        sh ../.version.sh > $@

doc-libshogun:	lib/versionstring.h
	-@$(SEDMI) "s/PROJECT_NUMBER         = .*/PROJECT_NUMBER         = `\
		sed -n 's/^#define VERSION_RELEASE "\(.*\)_"/\1/p' $@`/" ../../doc/Doxyfile
	cd ../../doc && $(MAKE)

doc-libshogunui:
	true

doc-elwms:
	true

doc-octave:
	echo "/** \page staticoctave Static Matlab/Octave Interface Function Reference" >x
	-echo "sg('set_output', 'y');sg('help','doxygen');" | \
		( $(LIBRARY_PATH)=../libshogunui:../libshogun \
		OCTAVE_LOADPATH="$(DESTDIR)$(OCTDIR)" octave )
	echo "*/" >>y
	cat x y >../../doc/pages/StaticOctaveInterface.mainpage
	rm -f x y

doc-cmdline:
	echo "/** \page staticcmdline Static Command Line Interface Function Reference" >x
	-bash -c "echo -e \"set_output y\nhelp doxygen\n\"" | \
		( $(LIBRARY_PATH)=../libshogunui:../libshogun  \
		PATH="./:$(DESTDIR)$(BINDIR)" shogun )
	echo "*/" >>y
	cat x y >../../doc/pages/StaticCommandLineInterface.mainpage
	rm -f x y

doc-python:
	echo "/** \page staticpython Static Python Interface Function Reference" >x
	-bash -c "echo -e \"from sg import sg\nsg('set_output', 'y')\nsg('help','doxygen')\n\"" | \
		( $(LIBRARY_PATH)=../libshogunui:../libshogun  PYTHONPATH="$(DESTDIR)$(PYDIR)" python )
	echo "*/" >>y
	cat x y >../../doc/pages/StaticPythonInterface.mainpage
	rm -f x y

doc-r:
	echo "/** \page staticr Static R Interface Function Reference" >x
	-echo "library(sg);sg('set_output', 'y');sg('help','doxygen');" | \
		( $(LIBRARY_PATH)=../libshogunui:../libshogun  R_PATH="$(DESTDIR)$(RDIR)" R --no-save)
	echo "*/" >>y
	cat x y >../../doc/pages/StaticRInterface.mainpage
	rm -f x y

doc-matlab:
	@echo "Check the octave documentation."
	@echo
	@echo " or just use sg('help') "
	@echo

doc-octave_modular:
	true

doc-python_modular:
	true

doc-r_modular:
	true

$(OBJFILES): ../.config .depend Makefile ../ChangeLog

$(LIBSHOGUN_TARGET): .depend $(OBJFILES) $(SRCFILES) $(HEADERFILES) $(OTHERDEPS)
	$(LINK) $(shell find $(SRCDIR) -name "*.$(EXT_OBJ_CPP)" -o \
		-name "*.$(EXT_OBJ_C)" 2>/dev/null) $(LINKFLAGS) -o $@ $(POSTLINKFLAGS)
	@$(POSTLINKCMD) $@
	@$(LIBSYMLINKCMD)

libshogun.a: .depend $(OBJFILES) $(SRCFILES) $(HEADERFILES)
	-ar rcs $@ $(shell find $(SRCDIR) -name "*.$(EXT_OBJ_CPP)" -o \
		-name "*.$(EXT_OBJ_C)" 2>/dev/null)

libshogunui.a: .depend $(OBJFILES) $(SRCFILES) $(HEADERFILES)
	-ar rcs $@ $(shell find $(SRCDIR) -name "*.$(EXT_OBJ_CPP)" -o \
		-name "*.$(EXT_OBJ_C)" 2>/dev/null)

%.R: %.${EXT_LIB_SWIG_LIBSHOGUN}
%.RData: %.R
	echo 'f="$<"; fdata="$@"; source(f); save(list=ls(all=TRUE),file=fdata, compress=TRUE); q(save="no")' | \
		($(LIBRARY_PATH)=../libshogun:../libshogunui R --no-save)

$(PRE_LIB_SWIG_LIBSHOGUN)%.${EXT_LIB_SWIG_LIBSHOGUN}:	%$(EXT_OBJ_SWIG)
	$(LINK) $(LINKFLAGS) -o $@ $^ $(POSTLINKFLAGS)

%${EXT_OBJ_SWIG}:	%$(EXT_SRC_SWIG)
	$(COMP_CPP) $(COMPFLAGS_SWIG_CPP) $(DEFINES) -c $(INCLUDES) -o $@ $<

%.$(EXT_OBJ_CPP):	%.$(EXT_SRC_CPP)
	$(COMP_CPP) $(COMPFLAGS_CPP) $(DEFINES) -c $(INCLUDES) -o $@ $<

%.$(EXT_OBJ_C):	%.$(EXT_SRC_C)
	$(COMP_C) $(COMPFLAGS_C) $(DEFINES) -c $(INCLUDES) -o $@ $<

ifeq (yes,$(USE_DOXYGEN))
%${EXT_SRC_SWIG}:	%$(EXT_SWIG_DOXY) %.$(EXT_INP_SWIG)
	$(SWIG) $(SWIGFLAGS) $(DEFINES_SWIG) $(DEFINES) $*.$(EXT_INP_SWIG)
	$(PYTHON) ../.scrub_docstrings.py $(EXT_IF_SWIG_LIBSHOGUN) $@

%$(EXT_SWIG_DOXY): %$(EXT_DOXY)
	$(DOXYGEN) $<
	$(PYTHON) ../.doxy2swig.py --quiet --no-function-definition \
		$(shell basename $< $(EXT_DOXY))/doxygen_xml/index.xml $@
else
%${EXT_SRC_SWIG}: %.$(EXT_INP_SWIG)
	$(SWIG) $(SWIGFLAGS) $(DEFINES_SWIG) $(DEFINES) $*.$(EXT_INP_SWIG)
	$(PYTHON) ../.scrub_docstrings.py $(EXT_IF_SWIG_LIBSHOGUN) $@
endif

# Do not include dependencies when they are about to be removed anyway.
ifneq ($(MAKECMDGOALS),distclean)
ifneq ($(MAKECMDGOALS),clean)

# Generate dependecies.
.depend:	../.config $(DEPDEPS)
ifeq (makedepend,$(MAKEDEPEND))
	-$(MAKEDEPEND) $(DEFINES) $(INCLUDES) \
		$(shell find $(SRCDIR) -name "*.$(EXT_SRC_CPP)" ) $(MAKEDEPENDOPTS) | \
		sed -e 's/\.o:/.$(EXT_OBJ_CPP):/g' >.depend
	-$(MAKEDEPEND) $(DEFINES) $(INCLUDES) \
		$(shell find $(SRCDIR) -name "*.$(EXT_SRC_C)" ) $(MAKEDEPENDOPTS) | \
		sed -e 's/\.o:/.$(EXT_OBJ_C):/g' >>.depend
else #gcc -MM
	find $(SRCDIR) -name "*.$(EXT_SRC_C)" -o -name "*.$(EXT_SRC_CPP)" \
	-exec $(MAKEDEPEND) $(DEFINES) $(INCLUDES) {} -MT {}.o \; >.depend
	( find . -name "*.$(EXT_INP_SWIG)" | cut -c 3- ; find ../shogun/ -name "*.h"  ) | \
	 		sed 's#/[/]*#/#g' | \
			$(PYTHON) ../.generate_link_dependencies.py "$(PRE_LIB_SWIG_LIBSHOGUN)" \
			"$(EXT_LIB_SWIG_LIBSHOGUN)" $(LIBSHOGUN_MODE) >>.depend
ifeq (r_modular,$(LIBSHOGUN_TARGET))
	    bash -c "( for i in \`grep '.$(EXT_LIB):' .depend | cut -f 1 -d ':'\` ; \
			do echo \`echo \$$i | sed s/.$(EXT_LIB)/.R:/\` \$$i ; done ) >>.depend"
endif

endif

# Include dependecies.
-include .depend
endif
endif
