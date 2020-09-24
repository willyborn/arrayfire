/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if !defined(AF_CPU)

#include <common/kernel_cache.hpp>

#include <common/compile_module.hpp>
#include <common/util.hpp>
#include <device_manager.hpp>
#include <platform.hpp>

#include <algorithm>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

using detail::Kernel;
using detail::Module;

using std::back_inserter;
using std::shared_timed_mutex;
using std::string;
using std::transform;
using std::unordered_map;
using std::vector;

namespace common {

using ModuleMap = unordered_map<size_t, Module>;

shared_timed_mutex& getCacheMutex(const int device) {
    static shared_timed_mutex mutexes[detail::DeviceManager::MAX_DEVICES];
    return mutexes[device];
}

ModuleMap& getCache(const int device) {
    static ModuleMap* caches =
        new ModuleMap[detail::DeviceManager::MAX_DEVICES];
    return caches[device];
}

Module findModule(const int device, const size_t& key) {
    std::shared_lock<shared_timed_mutex> readLock(getCacheMutex(device));
    auto& cache = getCache(device);
    auto iter   = cache.find(key);
    if (iter != cache.end()) { return iter->second; }
    return Module{};
}

Kernel getKernel(const string& kernelName, const vector<string>& sources,
                 const vector<TemplateArg>& targs,
                 const vector<string>& options, const size_t hashSources, const bool sourceIsJIT) {

    string tInstance = kernelName;
	auto targsIt = targs.begin();
	auto targsEnd = targs.end();
	if (targsIt != targsEnd) {
		tInstance += '<' + targsIt->_tparam;
		while (++targsIt != targsEnd) { tInstance += ',' + targsIt->_tparam; }
		tInstance += '>';
	}

    const bool notJIT = !sourceIsJIT;
	size_t hash = 0;
	if (notJIT) {
		hash = hashSources ? hashSources : deterministicHash(sources, hash);
		hash = deterministicHash(options, hash);
	};
    const size_t moduleKey = deterministicHash(tInstance, hash);
    const int device       = detail::getActiveDeviceId();
    Module currModule      = findModule(device, moduleKey); 
	if (!currModule) {
		currModule = loadModuleFromDisk(device, moduleKey, sourceIsJIT);
		if (!currModule) {
			currModule = compileModule(moduleKey, sources, options, { tInstance },
				sourceIsJIT);
		}
		std::unique_lock<shared_timed_mutex> writeLock(getCacheMutex(device));
		auto& cache = getCache(device);
		auto iter = cache.find(moduleKey);
		if (iter == cache.end()) {
			// If not found, this thread is the first one to compile this
			// kernel. Keep the generated module.
			Module mod = currModule;
			getCache(device).emplace(moduleKey, mod);
		}
		else {
			currModule.unload();  // dump the current threads extra compilation
			currModule = iter->second;
		}
	}

#if defined(AF_CUDA)
    return getKernel(currModule, tInstance, sourceIsJIT);
#elif defined(AF_OPENCL)
    return getKernel(currModule, kernelName, sourceIsJIT);
#endif
}

}  // namespace common

#endif
