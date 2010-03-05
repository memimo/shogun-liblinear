# Usage scenarios
#
# * To make a release (and tag it) run
#
#       make prepare-release
#       make svn-tag-release  
#       (cd releases/shogun_X.Y.Z ; make release ; make update-webpage)
#
# * To create a debian .orig.tar.gz run
#
#       make release DEBIAN=yes
#
# * To create a snapshot run
#
#       make release SNAPSHOT=yes
# 
# * To sign/md5sum the created tarballs and copy them to the webpage
#
#       make update-webpage
#
#
# The following additional options may be set
#
# DEBIAN=yes   						-> use debian naming scheme shogun_0.1.1+svn1337.orig.tar.gz
# SVMLIGHT=no						-> remove svm light
# COMPRESS=cruncher					-> use cruncher (bz2/gz etc as file compressor)
# SNAPSHOT=yes						-> use svn snapshot naming scheme
# MAINVERSION=0.2.0					-> main version
# EXTRAVERSION=+svn20061202			-> extra version string
# RELEASENAME=shogun-3.0.0+extra	-> use different releasename, here shogun-3.0.0+extra
#
# * For example to use gzip instead of bzip2 and to append an extra version
# string.
#       make release COMPRESS=gzip EXTRAVERSION=+svn20061202
#
# * To create a debian snapshot package
#
#      make DEBIAN=yes SNAPSHOT=yes
#

DEBIAN := no
SVMLIGHT := yes
COMPRESS := bzip2
MAINVERSION := $(shell awk '/Release/{print $$5;exit}' src/NEWS)
VERSIONBASE := $(shell echo $(MAINVERSION) | cut -f 1-2 -d '.')
EXTRAVERSION := 
RELEASENAME := shogun-$(MAINVERSION)$(EXTRAVERSION)
SVNVERSION := $(shell svn info 2>/dev/null | grep Revision: | cut -d ' ' -f 2)
LOGFILE := 'prepare-release.log'

stop:
	@echo
	@echo "To install shogun, please go to the 'src' directory,"
	@echo "most of the time"
	@echo
	@echo "cd src"
	@echo "./configure"
	@echo "make"
	@echo
	@echo "which will build all available interfaces of shogun"
	@echo "should be enough. For further information consult"
	@echo "especially the INSTALL and README files and try"
	@echo
	@echo "./configure --help"
	@echo

debian-package:	DEBIAN=yes
ifeq ($(DEBIAN),yes)
COMPRESS := gzip
SVMLIGHT := no
RELEASENAME := shogun_$(MAINVERSION)
ifeq ($(SNAPSHOT),yes)
RELEASENAME := $(RELEASENAME)+svn$(SVNVERSION)
endif
RELEASENAME := $(RELEASENAME).orig
else
ifeq ($(SNAPSHOT),yes)
RELEASENAME := $(RELEASENAME)+svn$(SVNVERSION)
endif
all: doc release matlab python octave
endif


.PHONY: all release package-from-release update-webpage svn-ignores clean distclean

DESTDIR := ../$(RELEASENAME)
REMOVE_SVMLIGHT := rm -f $(DESTDIR)/src/libshogun/classifier/svm/SVM_light.* $(DESTDIR)/src/libshogun/classifier/svm/Optimizer.* $(DESTDIR)/src/libshogun/regression/svr/SVR_light.* $(DESTDIR)/src/LICENSE.SVMlight; \
rm -f $(DESTDIR)/testsuite/data/classifier/SVMLight* $(DESTDIR)/testsuite/data/regression/SVRLight*	; \
grep -rl USE_SVMLIGHT $(DESTDIR)| xargs --no-run-if-empty sed -i '/\#ifdef USE_SVMLIGHT/,/\#endif \/\/USE_SVMLIGHT/c \\' ; \
sed -i '/^ \* EXCEPT FOR THE KERNEL CACHING FUNCTIONS WHICH ARE (W) THORSTEN JOACHIMS/,/ \* this program is free software/c\ * This program is free software; you can redistribute it and/or modify' $(DESTDIR)/src/libshogun/kernel/Kernel.cpp $(DESTDIR)/src/libshogun/kernel/Kernel.h ; \
sed -i '/^SVMlight:$$/,/^$$/c\\' $(DESTDIR)/src/LICENSE

# We assume that a release is always created from a SVN working copy.

prepare-release:
	@if [ -f $(LOGFILE) ]; then rm -f $(LOGFILE); fi
	svn update | tee --append $(LOGFILE)
	#update changelog
	svn stat | tee --append $(LOGFILE)
	@echo | tee --append $(LOGFILE)
	@echo "Please check output of 'svn stat'." | tee --append $(LOGFILE)
	@echo "Press ENTER to continue or Ctrl-C to abort." | tee --append $(LOGFILE)
	@read foobar
	(cd src;  rm -f ChangeLog ; $(MAKE) ChangeLog ; svn ci -m "updated changelog") | tee --append $(LOGFILE)
	#build for all interfaces and update doc
	+$(MAKE) -C src distclean | tee --append $(LOGFILE)
	( cd src && ./configure ) | tee --append $(LOGFILE)
	+$(MAKE) -C src | tee --append $(LOGFILE)
	+$(MAKE) -C src install DESTDIR=/tmp/sg_test_build | tee --append $(LOGFILE)
	+$(MAKE) -C src doc DESTDIR=/tmp/sg_test_build | tee --append $(LOGFILE)
	+$(MAKE) -C src tests DESTDIR=/tmp/sg_test_build | tee --append $(LOGFILE)
	+$(MAKE) -C src distclean | tee --append $(LOGFILE)
	(cd doc; svn ci -m "updated reference documentation") | tee --append $(LOGFILE)

release: src/libshogun/lib/versionstring.h $(DESTDIR)/src/libshogun/lib/versionstring.h
	tar -c -f $(DESTDIR).tar -C .. $(RELEASENAME)
	rm -f $(DESTDIR).tar.bz2 $(DESTDIR).tar.gz
	$(COMPRESS) -9 $(DESTDIR).tar

svn-tag-release: src/libshogun/lib/versionstring.h
	sed -i 's/VERSION_RELEASE "svn/VERSION_RELEASE "v$(MAINVERSION)/' src/libshogun/lib/versionstring.h
	sed -i "s/PROJECT_NUMBER         = .*/PROJECT_NUMBER         = v$(MAINVERSION)/" doc/Doxyfile
	svn ci -m "Preparing for new Release shogun_$(MAINVERSION)"
	#-cd .. && svn --force rm releases/shogun_$(MAINVERSION)
	#-cd .. && svn commit releases -m "clean old tag"
	svn cp https://svn.tuebingen.mpg.de:/shogun/trunk ../releases/shogun_$(MAINVERSION)
	cp src/libshogun/lib/versionstring.h ../releases/shogun_$(MAINVERSION)/src/libshogun/lib/versionstring.h
	sed -i "s| lib/versionstring.h||" ../releases/shogun_$(MAINVERSION)/src/Makefile
	cd ../releases && svn add shogun_$(MAINVERSION)/src/libshogun/lib/versionstring.h
	cd ../releases && svn ci -m "Tagging shogun_$(MAINVERSION) release"

package-from-release:
	rm -rf $(DESTDIR)
	mkdir $(DESTDIR)
	find ./ -regextype posix-egrep  ! -regex '(.*svn.*|.*\.o$$|.*wrap.*|.*\.so$$|.*\.mat$$|.*\.pyc$$|.*\.log$$)' \
		| xargs tar --no-recursion -cf - | tar -C $(DESTDIR) -xf - 
	if test ! $(SVMLIGHT) = yes; then $(REMOVE_SVMLIGHT); fi
	tar -c -f $(DESTDIR).tar -C .. $(RELEASENAME)
	rm -f $(DESTDIR).tar.bz2 $(DESTDIR).tar.gz
	$(COMPRESS) -9 $(DESTDIR).tar

update-webpage: 
	md5sum $(DESTDIR).tar.bz2 >$(DESTDIR).md5sum
	gpg --sign $(DESTDIR).tar.bz2

	ssh vserver mkdir -m 0755 -p /pub/shogun-ftp/releases/$(VERSIONBASE)/sources
	scp $(DESTDIR).tar.bz2 $(DESTDIR).tar.bz2.gpg $(DESTDIR).md5sum \
		vserver:/pub/shogun-ftp/releases/$(VERSIONBASE)/sources/
	ssh vserver chmod 644 /pub/shogun-ftp/releases/$(VERSIONBASE)/sources/*.*
	
	rm -rf doc/html
	make -C doc
	ssh vserver rm -f /pub/shogun/doc/*.*
	cd doc/html && tar --exclude='*.map' --exclude='*.md5' -cjf - . | ssh vserver tar -C /pub/shogun/doc/ -xjvf -
	make -C doc doc_cn
	cd doc/html_cn && tar --exclude='*.map' --exclude='*.md5' -cjf - . | ssh vserver tar -C /pub/shogun/doc_cn/ -xjvf -
	ssh vserver chmod 644 /pub/shogun/doc*/*.*
	ssh vserver ./bin/shogun_doc_install.sh
	rm -rf doc/html*

	cd ../../website && make

src/libshogun/lib/versionstring.h:
	rm -f src/ChangeLog
	make -C src ChangeLog
	make -C src/libshogun lib/versionstring.h

$(DESTDIR)/src/libshogun/lib/versionstring.h: src/libshogun/lib/versionstring.h
	rm -rf $(DESTDIR)
	svn export . $(DESTDIR)
	if test ! $(SVMLIGHT) = yes; then $(REMOVE_SVMLIGHT); fi

	# remove top level makefile from distribution
	rm -f $(DESTDIR)/src/.authors
	cp -f src/libshogun/lib/versionstring.h $(DESTDIR)/src/libshogun/lib/

svn-ignores: .svn_ignores
	find . -name .svn -prune -o -type d -exec svn propset svn:ignore -F .svn_ignores {} \;

clean:
	rm -rf $(DESTDIR)

distclean:
	make -C src distclean
	rm -rf $(DESTDIR) $(DESTDIR).tar.bz2 $(DESTDIR).tar.gz
