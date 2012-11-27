#/bin/bash

wget https://spideroak.com/share/ONYGSZDFOJJWQYLSMU/jewtube_lobbyist/home/sado/confSync/GraphicsMagick-1.3.12.tar.gz

tar -xvf GraphicsMagick-1.3.12.tar.gz

GM='GraphicsMagick-1.3.12'

mv effect.c gpumod.cu gpumod.h $GM/magick/

cd $GM

./configure --prefix=$HOME/GMinst

cat Makefile | sed -e 's%utilities_gm_LDADD = $(LIBMAGICK)%utilities_gm_LDADD = $(LIBMAGICK) -lcudart -lcuda -lgpumod%' |\
sed -e 's%utilities_gm_LDFLAGS = $(LDFLAGS)%utilities_gm_LDFLAGS = $(LDFLAGS) -L/usr/local/cuda/lib64 -Lmagick/%' |\
sed -e 's%Tpo magick/$(DEPDIR)/magick_libGraphicsMagick_la-effect.Plo%Tpo magick/$(DEPDIR)/magick_libGraphicsMagick_la-effect.Plo\n\tnvcc -c magick/gpumod.cu -o magick/libgpumod.a\n%' > tmp

mv tmp Makefile

make
make install
