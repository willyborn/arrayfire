@SETLOCAL

@REM Update the arrayfire dependencies in vcpkg.json file accordingly:  "intel-mkl"  or "fftw3","openblas"
@REM SET "AF_COMPUTE_LIBRARY=Intel-MKL"
@SET "AF_COMPUTE_LIBRARY=FFTW/LAPACK/BLAS"

RD /S build
@REM RD /S /Q  build
@IF NOT EXIST build (MKDIR build) ELSE (DEL /Q build\CMakeCache.txt)
@CD build
@IF "%AF_COMPUTE_LIBRARY%" == "Intel-MKL" (
	CALL "%ONE_API%\setvars.bat"
	SET "MKL_OPTS=-DUSE_CPU_MKL -DUSE_OPENCL_MKL -DUSE_CUDA_MKL"
	)
@cmake -Wno-dev -G="Visual Studio 16 2019" -A=x64 ^
-DAF_BUILD_CPU=ON ^
-DAF_BUILD_CUDA=ON ^
-DAF_BUILD_OPENCL=ON ^
-DAF_COMPUTE_LIBRARY="%AF_COMPUTE_LIBRARY%" %MKL_OPTS% ^
-DCMAKE_CXX_FLAGS="/DWIN32 /D_WINDOWS /W3 /GR /EHsc /DCL_TARGET_OPENCL_VERSION=120" ^
-DCUDA_NVCC_FLAGS="-Wno-deprecated-gpu-targets" ^
-DCMAKE_TOOLCHAIN_FILE="%VCPKG_DIR%/scripts/buildsystems/vcpkg.cmake" .. 
