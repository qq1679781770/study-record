### 虚拟地址翻译成物理地址
---
#### 添加系统调用
`arch/x86/include/asm/syscalls.h`添加</br>`asmlinkage unsigned long  sys_get_physical_address(unsigned long);`

`arch/x86/entry/syscalls/syscall_64.tbl`添加</br>`437     64      get_physical_address    sys_get_physical_address`

`kernel/sys.c`添加</br>
```
asmlinkage  unsigned long  sys_get_physical_address(unsigned long vaddr){

    pgd_t *pgd;
    p4d_t *p4d;
    pud_t *pud;
    pmd_t *pmd;
    pte_t *pte;
    unsigned long paddr = 0;
    unsigned long page_addr = 0;
    unsigned long page_offset = 0;
    pgd = pgd_offset(current->mm, vaddr);
    printk("pgd_val = 0x%lx\n", pgd_val(*pgd));
    printk("pgd_index = %lu\n", pgd_index(vaddr));
    if (pgd_none(*pgd)) {
        printk("not mapped in pgd\n");
        return -1;
    }
    p4d = p4d_offset(pgd, vaddr);
    printk("p4d_val = 0x%lx\n", p4d_val(*p4d));
    if (p4d_none(*p4d)) {
        printk("not mapped in p4d\n");
        return -1;
    }
    pud = pud_offset(p4d, vaddr);
    printk("pud_val = 0x%lx\n", pud_val(*pud));
    if (pud_none(*pud)) {
        printk("not mapped in pud\n");
        return -1;
    }
    pmd = pmd_offset(pud, vaddr);
    printk("pmd_val = 0x%lx\n", pmd_val(*pmd));
    printk("pmd_index = %lu\n", pmd_index(vaddr));
    if (pmd_none(*pmd)) {
        printk("not mapped in pmd\n");
        return -1;
    }
    pte = pte_offset_map(pmd, vaddr);
    printk("pte_val = 0x%lx\n", pte_val(*pte));
    printk("pte_index = %lu\n", pte_index(vaddr));
    if (pte_none(*pte)) {
        printk("not mapped in pte\n");
        return -1;
    }
    page_addr = pte_val(*pte) & PAGE_MASK;
    page_offset = vaddr & ~PAGE_MASK;
    paddr = page_addr | page_offset;
    printk("page_addr = %lx, page_offset = %lx\n", page_addr, page_offset);
    printk("vaddr = %lx, paddr = %lx\n", vaddr, paddr);
    return paddr;
}
```

#### c程序
```
lude<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>
#include<sys/syscall.h>
#include<errno.h>
int main(){
	int* p=(int*)malloc(sizeof(int));
	*p = 4;
	unsigned long vaddr =(unsigned long)(&p);
	unsigned long a = syscall(437,vaddr);
        printf("%lx,%lx\n",a,vaddr);
	printf("%s\n",strerror(errno));

	free(p);
	return 0;
}
```

###### 运行结果
```
printk输出
[  245.966981] pgd_val = 0x7b571067
[  245.966985] pgd_index = 350
[  245.966986] p4d_val = 0x7b571067
[  245.966987] pud_val = 0x7b572067
[  245.966988] pmd_val = 0x237f5067
[  245.966988] pmd_index = 28
[  245.966990] pte_val = 0x800000000d4ae063
[  245.966990] pte_index = 507
[  245.966992] page_addr = 800000000d4ae000, page_offset = f58
[  245.966993] vaddr = ffffaf53039fbf58, paddr = 800000000d4aef58
程序输出
800000000d728f58,7ffd69a0deb0
Success
```




