#include <stdio.h>
#include <sys/mman.h>

enum Flag{
    BEST_FIT,WORST_FIT,FIRST_FIT,NEXT_FIT
};

struct node_t{
    int size;
    struct node_t* next;
};

struct head_t{
    int size;
    int magic;
};


class Alloc{
private:
    node_t* head;
    node_t* cur;
    node_t* cur_pre;
    /* data */
public:
    void* malloc(size_t size,Flag flag);
    void free(void* ptr);
    void print();
    Alloc(/* args */);
    ~Alloc();
};

Alloc::Alloc(/* args */){
    head = (node_t*)mmap(NULL,4096,PROT_READ|PROT_WRITE,MAP_ANON|MAP_PRIVATE,-1,0);
    head->size = 4096 - sizeof(node_t);
    head->next = NULL;
    cur = head;
    cur_pre = NULL;
}

Alloc::~Alloc(){
    if(head->size==(4096-sizeof(node_t))){
        munmap((void*)head,4096);
    }
}

void* Alloc::malloc(size_t size,Flag flag){
    node_t* p = head;
    node_t* pre = NULL;
    node_t* best_pre=NULL;
    node_t* best = NULL;
    node_t* worst_pre=NULL;
    node_t* worst = NULL;
    size_t demand_size = size+sizeof(head_t);
    switch (flag){    
    case BEST_FIT:      
        while(p){          
            if(p->size>=demand_size){
                if(best){
                    if(p->size<best->size){
                        best_pre = pre;
                        best = p;
                    }
                }else{
                    best_pre = pre;
                    best = p;
                }
            }
            pre = p;
            p = p->next;
        }
        pre = best_pre;
        p = best;
        break;
    case WORST_FIT:     
        while(p){          
            if(p->size>=demand_size){
                if(worst){
                    if(p->size>worst->size){
                        worst_pre = pre;
                        worst = p;
                    }
                }else{
                    worst_pre = pre;
                    worst = p;
                }
            }
            pre = p;
            p = p->next;
        }
        pre = worst_pre;
        p = worst;
        break;  
    case FIRST_FIT:       
        while (p&&p->size<demand_size){
            pre = p; 
            p = p->next;
        }
        break;
    case NEXT_FIT:
        while(cur&&cur->size<demand_size){
            cur_pre = cur;
            cur = cur->next;
        }
        if(cur==NULL){
            cur_pre = NULL;
            cur = head;
            while(cur&&cur->size<demand_size){
                cur_pre = cur;
                cur = cur->next;
            }           
        }
        p = cur;
        pre = cur_pre;
        break;
    default:
        break;
    }
    if(p){
        head_t* head_ptr=NULL;
        node_t* next = p->next;
        int size_ = p->size;
        head_ptr = (head_t*)p;
        head_ptr->size = size;
        head_ptr->magic = 1234567;
        p = (node_t*)((void*)p+demand_size);
        p->size = size_-demand_size;
        p->next = next;
        if(pre) pre->next = p;
        else head = p;
        void* res = (void*)head_ptr+sizeof(head_ptr);
        return res;
    }else return NULL;

}
void Alloc::free(void* ptr){
    head_t* head_ptr = (head_t*)(ptr-sizeof(head_t));

    if(head_ptr->magic!=1234567) return;
    head_ptr->magic = -1;
    int total_size = head_ptr->size+sizeof(head_t);
    node_t* item = (node_t*)head_ptr;
    item->size = total_size-sizeof(node_t);
    node_t* pre = NULL;
    node_t* p = head;
    while(p&&item>p){
        pre = p;
        p = p->next;
    }
    if(p){
        if(pre){
            if(((void*)pre+sizeof(node_t)+pre->size)==(void*)item){
                if(((void*)item+total_size)==(void*)p){
                    pre->size = pre->size+total_size+sizeof(node_t)+p->size;
                    pre->next = p->next;
                }else{
                    pre->size = pre->size+total_size;
                    pre->next = p;
                }
            }else if(((void*)item+total_size)==(void*)p){
                item->size = item->size+sizeof(node_t)+p->size;
                item->next = p->next;  
                pre->next = item;
            }else{
                pre->next = item;
                item->next = p;
            }
        }else{
            if(((void*)item+total_size)==(void*)p){
                item->size = item->size+sizeof(node_t)+p->size;
                item->next = p->next;               
            }else{
                item->next = p;
            }
            head = item;
        }
    }else{
        if(((void*)pre+sizeof(node_t)+pre->size)==(void*)item){
            pre->size = pre->size+total_size;
        }else{
            pre->next = item;
        }
    }
}
void Alloc::print(){
node_t* p = head;
   while(p){
       printf("%d,%u,%u\n",p->size,p,&(p->size));       
       p = p->next;
   }
}
void test_first_fit(){
    Alloc alloc;
    void* ptr1 = alloc.malloc(100,FIRST_FIT);
    void* ptr2 = alloc.malloc(100,FIRST_FIT);
    void* ptr3 = alloc.malloc(100,FIRST_FIT);
    printf("%d\n",sizeof(head_t));
    printf("%u\n",ptr1);
    printf("%u\n",ptr2);
    printf("%u\n",ptr3);
    alloc.print();
    alloc.free(ptr2);
    printf("----\n");
    alloc.print();
    //printf("%u\n",ptr2);
    alloc.free(ptr3);
    printf("----\n");
    alloc.print();
    alloc.free(ptr1);
    printf("----\n");
    alloc.print();
}

int main(){
    test_first_fit();
    return 0;
}