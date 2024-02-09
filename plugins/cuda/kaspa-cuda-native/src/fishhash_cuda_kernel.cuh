#include "fishhash_cuda_kernel.h"
#include "keccak.cuh"

#define FNV_PRIME 0x01000193

//change these in #define
//static int full_dataset_item_parents = 512;
#define full_dataset_item_parents 512
//static int num_dataset_accesses = 32;
#define num_dataset_accesses 32
//static int light_cache_rounds = 3;
#define light_cache_rounds 3

const int light_cache_num_items = 1179641;
//#define light_cache_num_items 1179641
const int full_dataset_num_items = 37748717;
//#define full_dataset_num_items 37748717

#define DEV_INLINE __device__ __forceinline__

#define copy(dst, src, count)        \
    for (int i = 0; i != count; ++i) \
    {                                \
        (dst)[i] = (src)[i];         \
    }


static DEV_INLINE uint32_t fnv1(uint32_t u, uint32_t v) noexcept {
    return (u * FNV_PRIME) ^ v;
}

DEV_INLINE hash512 fnv1(const hash512& u, const hash512& v) noexcept {
    hash512 r;
    for (size_t i = 0; i < sizeof(r) / sizeof(r.word32s[0]); ++i)
        r.word32s[i] = fnv1(u.word32s[i], v.word32s[i]);
    return r;
}

typedef struct item_state
	{
	    const hash512* const cache;
	    const int64_t num_cache_items;
	    const uint32_t seed;

	    hash512 mix;

	    DEV_INLINE item_state(const fishhash_context& ctx, int64_t index) noexcept
	      : cache{ctx.light_cache},
		    num_cache_items{ctx.light_cache_num_items},
		    seed{static_cast<uint32_t>(index)} {
			//printf("item_state debug 1 %p - %d", &cache, num_cache_items);
		    mix = cache[index % num_cache_items];
			//printf("item_state debug 2");
		    mix.word32s[0] ^= seed;
		    //keccak(mix.word64s, 512, mix.bytes, 64);
			//printf("item_state debug 3");
            SHA3_512(mix.uint2s);
	    }

	    DEV_INLINE void update(uint32_t round) noexcept {
		    static constexpr size_t num_words = sizeof(mix) / sizeof(uint32_t);
		    const uint32_t t = fnv1(seed ^ round, mix.word32s[round % num_words]);
		    const int64_t parent_index = t % num_cache_items;
		    mix = fnv1(mix, cache[parent_index]);
	    }

	    DEV_INLINE hash512 final() noexcept { 
	    	//keccak(mix.word64s, 512, mix.bytes, 64);
            SHA3_512(mix.uint2s);
	    	return mix; 
	    }
	} item_state;



DEV_INLINE hash1024 calculate_dataset_item_1024(const fishhash_context& ctx, uint32_t index) noexcept {
	//printf("heavy_hash Thread %d, Block %d\n", threadIdx.x, blockIdx.x);
	//printf("calculate_dataset_item_1024 debug 1");
    item_state item0{ctx, int64_t(index) * 2};
	//printf("calculate_dataset_item_1024 debug 2");
    item_state item1{ctx, int64_t(index) * 2 + 1};

	//printf("calculate_dataset_item_1024 debug 3");
	for (uint32_t j = 0; j < full_dataset_item_parents; ++j) {
		item0.update(j);
		item1.update(j);
	}

    hash512 it0 = item0.final();
    hash512 it1 = item1.final();

    return hash1024{{it0, it1}};
}

DEV_INLINE hash1024 lookup(const fishhash_context& ctx, uint32_t index) {
    if (ctx.full_dataset != NULL) {
		//printf("lookup debug 1");
        hash1024 * item = &ctx.full_dataset[index];
        
        // Ability to handle lazy lookup
        if (item->word64s[0] == 0) {
            *item = calculate_dataset_item_1024(ctx, index);
        }
        
        return *item;
    } else {
		//printf("lookup debug 2");
        return calculate_dataset_item_1024(ctx, index);
    }
}

DEV_INLINE hash256 fishhash_kernel( const fishhash_context& ctx, const hash512& seed) noexcept {
		//printf("fishhash_kernel debug 1");
		const uint32_t index_limit = static_cast<uint32_t>(ctx.full_dataset_num_items);
		//printf("fishhash_kernel debug 1.1");
		//const uint32_t seed_init = seed.word32s[0];
	    //printf("fishhash_kernel debug 2");
		hash1024 mix{seed, seed};
		//printf("fishhash_kernel debug 3");
		//printf("The index_limit is : %d \n", index_limit);
		for (uint32_t i = 0; i < num_dataset_accesses; ++i) {
					
			//printf("fishhash_kernel debug 4, %d", index_limit);
			//printf("fishhash_kernel debug 4.1, %032x", mix.word32s[0]);
			// Calculate new fetching indexes
			const uint32_t p0 = mix.word32s[0] % index_limit;
			//printf("fishhash_kernel debug 4.2, %032x", mix.word32s[4]);
			const uint32_t p1 = mix.word32s[4] % index_limit;
			//printf("fishhash_kernel debug 4.3, %032x", mix.word32s[8]);
			const uint32_t p2 = mix.word32s[8] % index_limit;
			
			//printf("fishhash_kernel debug 5");
			hash1024 fetch0 = lookup(ctx, p0);
			hash1024 fetch1 = lookup(ctx, p1);
			hash1024 fetch2 = lookup(ctx, p2);

			//printf("fishhash_kernel debug 6");
			// Modify fetch1 and fetch2
			for (size_t j = 0; j < 32; ++j) {
				fetch1.word32s[j] = fnv1(mix.word32s[j], fetch1.word32s[j]);
				fetch2.word32s[j] = mix.word32s[j] ^ fetch2.word32s[j];
			}

			//printf("fishhash_kernel debug 7");
	     	// Final computation of new mix
			for (size_t j = 0; j < 16; ++j)
				mix.word64s[j] = fetch0.word64s[j] * fetch1.word64s[j] + fetch2.word64s[j];
		}

		//printf("fishhash_kernel debug 8");
		// Collapse the result into 32 bytes
		hash256 mix_hash;
		static constexpr size_t num_words = sizeof(mix) / sizeof(uint32_t);
		//printf("fishhash_kernel debug 9");
		for (size_t i = 0; i < num_words; i += 4) {
			const uint32_t h1 = fnv1(mix.word32s[i], mix.word32s[i + 1]);
			const uint32_t h2 = fnv1(h1, mix.word32s[i + 2]);
			const uint32_t h3 = fnv1(h2, mix.word32s[i + 3]);
			mix_hash.word32s[i / 4] = h3;
		}

		//printf("fishhash_kernel debug 10");
		return mix_hash;
	}

DEV_INLINE void printHash(char* msg, const uint8_t* hash, int size) {
		printf(msg);
		for(int i = 0; i < size; i++) {
			//printf("%02x", output[i]);
			printf("%02x", hash[i]);
		}
		printf("\n");
	}

//DEV_INLINE void hashFish(uint8_t * output, const fishhash_context * ctx, const uint8_t * header, uint64_t header_size) noexcept {
DEV_INLINE void hashFish(
            const fishhash_context * ctx,
            uint8_t* out,
            const uint8_t* in) {
		hash512 seed; 
        //*seed.bytes = *in;
		memset(seed.bytes, 0, 64);
		memcpy(seed.bytes, in, 32);
		/*
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			printHash("hashFish-1 in is : ", in, 32);
			printHash("hashFish-1 in is : ", seed.bytes, 32);
		}
		*/
		//printf("hashFish debug 1");
		const hash256 mix_hash = fishhash_kernel(*ctx, seed);
	    //*out = *mix_hash.bytes;
		
		memcpy(out, mix_hash.bytes, 32);
		/*
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			printHash("hashFish-2 in is : ", mix_hash.bytes, 32);
			printHash("hashFish-2 in is : ", out, 32);
		}
		*/
	}



DEV_INLINE hash512 bitwise_xor(const hash512& x, const hash512& y) noexcept {
		hash512 z;
		for (size_t i = 0; i < sizeof(z) / sizeof(z.word64s[0]); ++i)
			z.word64s[i] = x.word64s[i] ^ y.word64s[i];
		return z;
	}

