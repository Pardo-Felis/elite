#include"stdio.h"
#include"stdlib.h"
#include"string.h"
#include"stdint.h"

#define ARRAY_SIZE 1024*1024
#define TIMES   1

uint32_t src0[ARRAY_SIZE];
uint32_t src1[ARRAY_SIZE];
uint32_t dest[ARRAY_SIZE+1];

#define COMPUTE_KERNEL() \
do\
{\
    uint32_t temp=0;\
    for(int i=0;i<1024;i++)\
    {\
        for(int j=0;j<1024;j++)\
        {\
            temp=0;\
            for(int k=0;k<1024;k++)\
            {\
                temp+=src0[i*1024+k]*src1[j*1024+k];\
            }\
            dest[i*1024+j]=temp;\
        }\
    }\
}\
while(0)
uint64_t rdtsc(void)
{
    uint32_t lo,hi;
    __asm__ __volatile__ ("rdtsc":"=a"(lo),"=d"(hi));
    return ((uint64_t)hi<<32)|lo;
}
uint32_t checksum(void)
{
    uint32_t final =0;

    for(int i=0;i<ARRAY_SIZE;i++) final+=dest[i];

    return final;
}
void __attribute__ ((optimize("00"))) raw_calc_naive(void)
{
    COMPUTE_KERNEL();
}
void __attribute__ ((optimize("02"))) raw_calc_expert(void)
{
    COMPUTE_KERNEL();
}
void __attribute__ ((optimize("03"))) raw_calc_sse(void)
{
    COMPUTE_KERNEL();
}
void __attribute__ ((optimize("03"),__target__("arch=core-avx2"))) raw_calc_avx_auto(void)
{
    COMPUTE_KERNEL();
}
void __attribute__ ((optimize("03"),__target__("arch=core-avx2"))) raw_calc_avx_manual(void)
{
    const uint32_t const0=0;
    const uint64_t limit=ARRAY_SIZE*32;
    const uint32_t nums=1024*32;
    /*No unroll
    __asm__ __volatile__
    (
        "xor            %%rcx,          %%rcx\n"                //rcx contains the counter
        "lea            %[src0],        %%rbx\n"                //rbx/rsi contains src
        "lea            %[src1],        %%rsi\n"
        "lea            %[dest],        %%rdi\n"                //rdi contains dest
        "vpbroadcastd   %[const0],      %%ymm13\n"              //ymm13,14and15 contains the constants
        "vpbroadcastd   %[const1],      %%ymm14\n"
        "vpbroadcastd   %[const2],      %%ymm15\n"
        "l:\n"
        "vpmulld        (%%rbx,%%rcx,1),%%ymm13,    %%ymm0\n"   //temp =src0[i]*0x12345678;
        "vmovdqu        (%%rsi,%%rcx,1),%%ymm1\n"               //temp +=src1[i]*0x76543210;
        "vpmulld        %%ymm1,         %%ymm14,    %%ymm2\n"
        "vpaddd         %%ymm0,         %%ymm2,     %%ymm2\n"
        "vpmulld        %%ymm2,         %%ymm15,    %%ymm2\n"   //temp *= 0xA0A00505;
        "vpaddd         %%ymm2,         %%ymm1,     %%ymm2\n"   //dest[i]=temp + src1[i]
        "vmovdqu        %%ymm2,         (%%rdi,%%rcx,1)\n"
        "add            $0x20,          %%rcx\n"
        "cmp            %%rcx,          %[limit]\n"
        "jne            lb\n"
        :[dest]"=m"(dest)
        :[src0]"m"(src0),[src1]"m"(src1),[const0]"m"(const0),[const1]"m"(const1),[const2]"m"(const2),[limit]"r"(limit)
        :"%rbx","%rcx","%rsi","%rdi","memory","cc"
    )*/

    /*Manual unroll with data dependency interleaving*/
    __asm__ __volatile__
    (
        "vpbroadcastd   %[const0],      %%ymm14\n"      //ymm13,14and15 contains the constant
        "xor            %%rcx,          %%rcx\n"                //rcx contains the counter
        "mov            $0x20,          %%rax\n"                //rax contains the second counter
        "mov            $0x40,          %%rdx\n"                //rdx contains the third counter
        "mov            $0x60,          %%rbp\n"                //rbp contains the fourth counter
        "mov            $0x0,          %%rdi\n"
        "r:\n"
        "lea            %[src0],        %%rbx\n"                //rbx/rsi contains src
        "lea            %[src1],        %%rsi\n"
        "vpbroadcastd   %[const0],      %%ymm12\n"              //ymm13,14and15 contains the constants
        "l:\n"
        //interleaved unroll -rcx/0,1,2 rax/3,4,5 rdx/6,7,8 rbp/9,10,11
        "vmovdqu        (%%rbx,%%rcx,1),%%ymm0\n"   //temp =src0[i]*0x12345678;
        "vmovdqu        (%%rbx,%%rax,1),%%ymm3\n"
        "vmovdqu        (%%rbx,%%rdx,1),%%ymm6\n"
        "vmovdqu        (%%rbx,%%rbp,1),%%ymm9\n"
        "vmovdqu        (%%rsi,%%rcx,1),%%ymm1\n"               //temp +=src1[i]*0x76543210;
        "vpmulld        %%ymm1,         %%ymm0,    %%ymm2\n"
        "vmovdqu        (%%rsi,%%rax,1),%%ymm4\n"
        "vpmulld        %%ymm4,         %%ymm3,    %%ymm5\n"
        "vmovdqu        (%%rsi,%%rdx,1),%%ymm7\n"
        "vpmulld        %%ymm7,         %%ymm6,    %%ymm8\n"
        "vmovdqu        (%%rsi,%%rbp,1),%%ymm10\n"
        "vpmulld        %%ymm10,        %%ymm9,    %%ymm11\n"

        "vpaddd         %%ymm2,         %%ymm5,     %%ymm5\n"
        "vpaddd         %%ymm8,         %%ymm11,    %%ymm11\n"

        "vpaddd         %%ymm5,         %%ymm12,    %%ymm12\n"

        "vpaddd         %%ymm11,        %%ymm12,    %%ymm12\n"

        "add            $0x80,          %%rcx\n"
        "add            $0x80,          %%rax\n"
        "add            $0x80,          %%rdx\n"
        "add            $0x80,          %%rbp\n"

        "cmp            %%rcx,          %[nums]\n"
        "jne            l\n"
        "lea            %[dest],        %%rbx\n"                //rdi contains dest
        "vmovdqu        %%ymm12,         (%%rbx,%%rdi,1)\n"

        "add            $0x20,          %%rdi\n"
        "cmp            %%rdi,          %[limit]\n"
        "jne            r\n"

        :[dest]"=m"(dest)
        :[src0]"m"(src0),[src1]"m"(src1),[const0]"m"(const0),[nums]"m"(nums),[limit]"r"(limit)
        :"%rax","%rbx","%rcx","%rdx","%rsi","%rdi","%rbp","memory","cc"
    );
}


void test(char* opt,void (*func)(void))
{
    uint64_t start;
    uint64_t end;
    memset(dest,0x00,ARRAY_SIZE*sizeof(uint32_t));
    start=rdtsc();
    FILE* pf = fopen("A1.txt", "r");
	for(int i=0;i<ARRAY_SIZE;i++) fscanf(pf, "%d",&src0[i]);
    fclose(pf);
    func();
    pf = fopen("A2.txt", "r");
    for(int i=0;i<ARRAY_SIZE;i++) fscanf(pf, "%d",&src0[i]);
    func();
    fclose(pf);
    pf = fopen("A3.txt", "r");
    for(int i=0;i<ARRAY_SIZE;i++) fscanf(pf, "%d",&src0[i]);
    func();
    fclose(pf);
    pf = fopen("A4.txt", "r");
    for(int i=0;i<ARRAY_SIZE;i++) fscanf(pf, "%d",&src0[i]);
    func();
    fclose(pf);
    pf = fopen("A5.txt", "r");
    for(int i=0;i<ARRAY_SIZE;i++) fscanf(pf, "%d",&src0[i]);
    func();
    fclose(pf);
    pf = fopen("A6.txt", "r");
    for(int i=0;i<ARRAY_SIZE;i++) fscanf(pf, "%d",&src0[i]);
    func();
    fclose(pf);
    pf = fopen("A7.txt", "r");
    for(int i=0;i<ARRAY_SIZE;i++) fscanf(pf, "%d",&src0[i]);
    func();
    fclose(pf);
    pf = fopen("A8.txt", "r");
    for(int i=0;i<ARRAY_SIZE;i++) fscanf(pf, "%d",&src0[i]);
    func();
    fclose(pf);
    pf = fopen("A9.txt", "r");
    for(int i=0;i<ARRAY_SIZE;i++) fscanf(pf, "%d",&src0[i]);
    func();
    fclose(pf);
    pf = fopen("A10.txt", "r");
    for(int i=0;i<ARRAY_SIZE;i++) fscanf(pf, "%d",&src0[i]);
    func();
    fclose(pf);
    pf = fopen("A11.txt", "r");
    for(int i=0;i<ARRAY_SIZE;i++) fscanf(pf, "%d",&src0[i]);
    func();
    fclose(pf);
    pf = fopen("A12.txt", "r");
    for(int i=0;i<ARRAY_SIZE;i++) fscanf(pf, "%d",&src0[i]);
    func();
    fclose(pf);
    pf = fopen("A13.txt", "r");
    for(int i=0;i<ARRAY_SIZE;i++) fscanf(pf, "%d",&src0[i]);
    func();
    fclose(pf);
    pf = fopen("A14.txt", "r");
    for(int i=0;i<ARRAY_SIZE;i++) fscanf(pf, "%d",&src0[i]);
    func();
    fclose(pf);
    pf = fopen("A15.txt", "r");
    for(int i=0;i<ARRAY_SIZE;i++) fscanf(pf, "%d",&src0[i]);
    func();
    fclose(pf);
    pf = fopen("A16.txt", "r");
    for(int i=0;i<ARRAY_SIZE;i++) fscanf(pf, "%d",&src0[i]);
    func();
    fclose(pf);
	pf = NULL;
    
    end=rdtsc();
    printf("%-12s- %-12lu - %12llu cycles\n",opt,checksum(),end-start);
}

int main(int argc,char *argv[]) 
{
    FILE* pf = fopen("B.txt", "r");
	if (pf == NULL)
	{
		printf("%s\n", strerror(errno));
		return 1;
	}
	for(int i=0;i<ARRAY_SIZE;i++)
    {
        int j=0;
        fscanf(pf, "%d",&src0[j]);
        j+=1024;
        if(j>ARRAY_SIZE) j=j%1024+1;
    }
    fclose(pf);
	pf = NULL;
    
    test("naive",raw_calc_naive);
    test("expert",raw_calc_expert);
    test("sse",raw_calc_sse);
    test("avx-auto",raw_calc_avx_auto);
    test("avx-manual",raw_calc_avx_manual);
    //test("avx512",raw_calc_avx512);

    return 0;
}

