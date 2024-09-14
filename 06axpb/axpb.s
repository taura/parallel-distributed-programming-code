	.file	"axpb.cc"
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC0:
	.string	"long int axpb_simd_m_nmn(axpb_options_t, float, float*, float)"
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC1:
	.string	"axpb.cc"
.LC2:
	.string	"m % L == 0"
.LC3:
	.string	"n % steps_inner == 0"
.LC4:
	.string	"opt.bs == 1"
	.text
	.p2align 4
	.globl	_Z15axpb_simd_m_nmn14axpb_options_tfPff
	.type	_Z15axpb_simd_m_nmn14axpb_options_tfPff, @function
_Z15axpb_simd_m_nmn14axpb_options_tfPff:
.LFB5656:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rdi, %r8
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rax
	movq	64(%rbp), %rdi
	andq	$-64, %rsp
	testb	$15, %al
	jne	.L14
	testb	$3, %dil
	jne	.L15
	cmpq	$1, 32(%rbp)
	jne	.L16
#APP
# 222 "axpb.cc" 1
	# axpb_simd_m_nmn: ax+b loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L8
	testq	%rax, %rax
	leaq	15(%rax), %rcx
	cmovns	%rax, %rcx
	sarq	$4, %rcx
	cmpq	$15, %rax
	jle	.L8
	xorl	%esi, %esi
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	.p2align 4,,10
	.p2align 3
.L9:
	movq	%r8, %rax
	xorl	%edx, %edx
	.p2align 4,,10
	.p2align 3
.L7:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	incq	%rdx
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rcx, %rdx
	jl	.L7
	addq	$4, %rsi
	cmpq	%rsi, %rdi
	jg	.L9
	vzeroupper
.L8:
#APP
# 230 "axpb.cc" 1
	# axpb_simd_m_nmn: ax+b loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L14:
	.cfi_restore_state
	leaq	.LC0(%rip), %rcx
	movl	$218, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC2(%rip), %rdi
	call	__assert_fail@PLT
.L16:
	leaq	.LC0(%rip), %rcx
	movl	$220, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
.L15:
	leaq	.LC0(%rip), %rcx
	movl	$219, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC3(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5656:
	.size	_Z15axpb_simd_m_nmn14axpb_options_tfPff, .-_Z15axpb_simd_m_nmn14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi1EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC5:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 1]"
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi1EEl14axpb_options_tfPff.str1.1,"aMS",@progbits,1
.LC6:
	.string	"c % L == 0"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi1EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi1EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi1EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi1EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi1EEl14axpb_options_tfPff:
.LFB5718:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC5(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5718:
	.size	_Z24axpb_simd_parallel_m_mnmILi1EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi1EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi1EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC7:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 1]"
	.section	.rodata._Z15axpb_simd_m_mnmILi1EEl14axpb_options_tfPff.str1.1,"aMS",@progbits,1
.LC8:
	.string	"m % (c * L) == 0"
	.section	.text._Z15axpb_simd_m_mnmILi1EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi1EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi1EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi1EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi1EEl14axpb_options_tfPff:
.LFB5717:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rdx
	movq	64(%rbp), %rcx
	andq	$-64, %rsp
	testb	$15, %dl
	jne	.L31
	cmpq	$1, 32(%rbp)
	jne	.L32
	testq	%rdx, %rdx
	leaq	15(%rdx), %rsi
	cmovns	%rdx, %rsi
	vbroadcastss	%xmm0, %zmm2
	sarq	$4, %rsi
	xorl	%eax, %eax
	vbroadcastss	%xmm1, %zmm1
	cmpq	$15, %rdx
	jg	.L22
	jmp	.L28
	.p2align 4,,10
	.p2align 3
.L23:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$1>: ax+c inner loop end
# 0 "" 2
#NO_APP
	incq	%rax
	cmpq	%rsi, %rax
	jge	.L28
.L22:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$1>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rcx, %rcx
	jle	.L23
	movq	%rax, %r8
	salq	$6, %r8
	vmovups	(%rdi,%r8), %zmm0
	xorl	%edx, %edx
.L24:
	incq	%rdx
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	cmpq	%rdx, %rcx
	jne	.L24
	vmovups	%zmm0, (%rdi,%r8)
	jmp	.L23
	.p2align 4,,10
	.p2align 3
.L28:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L31:
	.cfi_restore_state
	leaq	.LC7(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L32:
	leaq	.LC7(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5717:
	.size	_Z15axpb_simd_m_mnmILi1EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi1EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi1EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC9:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 1]"
	.section	.rodata._Z11axpb_simd_cILi1EEl14axpb_options_tfPff.str1.1,"aMS",@progbits,1
.LC10:
	.string	"opt.m == c * L"
	.section	.text._Z11axpb_simd_cILi1EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi1EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi1EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi1EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi1EEl14axpb_options_tfPff:
.LFB5716:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$16, 56(%rbp)
	jne	.L40
	cmpq	$1, 32(%rbp)
	jne	.L41
	movq	64(%rbp), %rax
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$1>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rax, %rax
	jg	.L42
.L36:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$1>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L42:
	.cfi_restore_state
	vmovups	(%rdi), %zmm2
	vbroadcastss	%xmm0, %zmm0
	vbroadcastss	%xmm1, %zmm1
	xorl	%edx, %edx
.L37:
	incq	%rdx
	vfmadd132ps	%zmm0, %zmm1, %zmm2
	cmpq	%rdx, %rax
	jne	.L37
	vmovups	%zmm2, (%rdi)
	vzeroupper
	jmp	.L36
.L41:
	leaq	.LC9(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
.L40:
	leaq	.LC9(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5716:
	.size	_Z11axpb_simd_cILi1EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi1EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi2EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC11:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 2]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi2EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi2EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi2EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi2EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi2EEl14axpb_options_tfPff:
.LFB5721:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC11(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5721:
	.size	_Z24axpb_simd_parallel_m_mnmILi2EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi2EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi2EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC12:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 2]"
	.section	.text._Z15axpb_simd_m_mnmILi2EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi2EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi2EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi2EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi2EEl14axpb_options_tfPff:
.LFB5720:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rax
	movq	64(%rbp), %rdx
	andq	$-64, %rsp
	testb	$31, %al
	jne	.L57
	cmpq	$1, 32(%rbp)
	jne	.L58
	testq	%rax, %rax
	movq	%rdi, %rcx
	leaq	15(%rax), %rdi
	cmovns	%rax, %rdi
	vbroadcastss	%xmm0, %zmm3
	sarq	$4, %rdi
	xorl	%esi, %esi
	vbroadcastss	%xmm1, %zmm2
	cmpq	$15, %rax
	jle	.L54
	.p2align 4,,10
	.p2align 3
.L52:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$2>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L49
	vmovups	(%rcx), %zmm1
	vmovups	64(%rcx), %zmm0
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L50:
	incq	%rax
	vfmadd132ps	%zmm3, %zmm2, %zmm1
	vfmadd132ps	%zmm3, %zmm2, %zmm0
	cmpq	%rax, %rdx
	jne	.L50
	vmovups	%zmm1, (%rcx)
	vmovups	%zmm0, 64(%rcx)
.L49:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$2>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$2, %rsi
	subq	$-128, %rcx
	cmpq	%rdi, %rsi
	jl	.L52
.L54:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L57:
	.cfi_restore_state
	leaq	.LC12(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L58:
	leaq	.LC12(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5720:
	.size	_Z15axpb_simd_m_mnmILi2EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi2EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi2EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC13:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 2]"
	.section	.text._Z11axpb_simd_cILi2EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi2EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi2EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi2EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi2EEl14axpb_options_tfPff:
.LFB5719:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$32, 56(%rbp)
	jne	.L66
	cmpq	$1, 32(%rbp)
	jne	.L67
	movq	64(%rbp), %rdx
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$2>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L62
	vmovaps	%xmm0, %xmm3
	vmovaps	%xmm1, %xmm2
	vmovups	64(%rdi), %zmm0
	vmovups	(%rdi), %zmm1
	vbroadcastss	%xmm3, %zmm3
	vbroadcastss	%xmm2, %zmm2
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L63:
	incq	%rax
	vfmadd132ps	%zmm3, %zmm2, %zmm1
	vfmadd132ps	%zmm3, %zmm2, %zmm0
	cmpq	%rax, %rdx
	jne	.L63
	vmovups	%zmm1, (%rdi)
	vmovups	%zmm0, 64(%rdi)
	vzeroupper
.L62:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$2>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L66:
	.cfi_restore_state
	leaq	.LC13(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L67:
	leaq	.LC13(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5719:
	.size	_Z11axpb_simd_cILi2EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi2EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi3EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC14:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 3]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi3EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi3EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi3EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi3EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi3EEl14axpb_options_tfPff:
.LFB5724:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC14(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5724:
	.size	_Z24axpb_simd_parallel_m_mnmILi3EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi3EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi3EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC15:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 3]"
	.section	.text._Z15axpb_simd_m_mnmILi3EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi3EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi3EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi3EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi3EEl14axpb_options_tfPff:
.LFB5723:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$3074457345618258603, %rdx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %r9
	movq	64(%rbp), %rcx
	movq	%r9, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%r9, %rdx
	sarq	$3, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	leaq	(%rax,%rax,2), %rax
	salq	$4, %rax
	cmpq	%rax, %r9
	jne	.L82
	cmpq	$1, 32(%rbp)
	jne	.L83
	testq	%r9, %r9
	leaq	15(%r9), %r8
	cmovns	%r9, %r8
	movq	%rdi, %rdx
	sarq	$4, %r8
	xorl	%esi, %esi
	vbroadcastss	%xmm0, %zmm4
	vbroadcastss	%xmm1, %zmm3
	cmpq	$15, %r9
	jle	.L79
	.p2align 4,,10
	.p2align 3
.L77:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$3>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rcx, %rcx
	jle	.L74
	vmovups	(%rdx), %zmm2
	vmovups	64(%rdx), %zmm1
	vmovups	128(%rdx), %zmm0
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L75:
	incq	%rax
	vfmadd132ps	%zmm4, %zmm3, %zmm2
	vfmadd132ps	%zmm4, %zmm3, %zmm1
	vfmadd132ps	%zmm4, %zmm3, %zmm0
	cmpq	%rax, %rcx
	jne	.L75
	vmovups	%zmm2, (%rdx)
	vmovups	%zmm1, 64(%rdx)
	vmovups	%zmm0, 128(%rdx)
.L74:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$3>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$3, %rsi
	addq	$192, %rdx
	cmpq	%r8, %rsi
	jl	.L77
.L79:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L82:
	.cfi_restore_state
	leaq	.LC15(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L83:
	leaq	.LC15(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5723:
	.size	_Z15axpb_simd_m_mnmILi3EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi3EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi3EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC16:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 3]"
	.section	.text._Z11axpb_simd_cILi3EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi3EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi3EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi3EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi3EEl14axpb_options_tfPff:
.LFB5722:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$48, 56(%rbp)
	jne	.L91
	cmpq	$1, 32(%rbp)
	jne	.L92
	movq	64(%rbp), %rdx
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$3>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L87
	vmovaps	%xmm0, %xmm4
	vmovaps	%xmm1, %xmm3
	vmovups	(%rdi), %zmm2
	vmovups	64(%rdi), %zmm1
	vmovups	128(%rdi), %zmm0
	vbroadcastss	%xmm4, %zmm4
	vbroadcastss	%xmm3, %zmm3
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L88:
	incq	%rax
	vfmadd132ps	%zmm4, %zmm3, %zmm2
	vfmadd132ps	%zmm4, %zmm3, %zmm1
	vfmadd132ps	%zmm4, %zmm3, %zmm0
	cmpq	%rax, %rdx
	jne	.L88
	vmovups	%zmm2, (%rdi)
	vmovups	%zmm1, 64(%rdi)
	vmovups	%zmm0, 128(%rdi)
	vzeroupper
.L87:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$3>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L91:
	.cfi_restore_state
	leaq	.LC16(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L92:
	leaq	.LC16(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5722:
	.size	_Z11axpb_simd_cILi3EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi3EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi4EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC17:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 4]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi4EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi4EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi4EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi4EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi4EEl14axpb_options_tfPff:
.LFB5727:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC17(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5727:
	.size	_Z24axpb_simd_parallel_m_mnmILi4EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi4EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi4EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC18:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 4]"
	.section	.text._Z15axpb_simd_m_mnmILi4EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi4EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi4EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi4EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi4EEl14axpb_options_tfPff:
.LFB5726:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rax
	movq	64(%rbp), %rcx
	andq	$-64, %rsp
	testb	$63, %al
	jne	.L107
	cmpq	$1, 32(%rbp)
	jne	.L108
	testq	%rax, %rax
	movq	%rdi, %rdx
	leaq	15(%rax), %rdi
	cmovns	%rax, %rdi
	vmovaps	%xmm0, %xmm2
	vmovaps	%xmm1, %xmm0
	sarq	$4, %rdi
	xorl	%esi, %esi
	vbroadcastss	%xmm2, %zmm1
	vbroadcastss	%xmm0, %zmm0
	cmpq	$15, %rax
	jle	.L104
	.p2align 4,,10
	.p2align 3
.L102:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$4>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rcx, %rcx
	jle	.L99
	vmovups	(%rdx), %zmm5
	vmovups	64(%rdx), %zmm4
	vmovups	128(%rdx), %zmm3
	vmovups	192(%rdx), %zmm2
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L100:
	incq	%rax
	vfmadd132ps	%zmm1, %zmm0, %zmm5
	vfmadd132ps	%zmm1, %zmm0, %zmm4
	vfmadd132ps	%zmm1, %zmm0, %zmm3
	vfmadd132ps	%zmm1, %zmm0, %zmm2
	cmpq	%rax, %rcx
	jne	.L100
	vmovups	%zmm5, (%rdx)
	vmovups	%zmm4, 64(%rdx)
	vmovups	%zmm3, 128(%rdx)
	vmovups	%zmm2, 192(%rdx)
.L99:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$4>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$4, %rsi
	addq	$256, %rdx
	cmpq	%rdi, %rsi
	jl	.L102
.L104:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L107:
	.cfi_restore_state
	leaq	.LC18(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L108:
	leaq	.LC18(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5726:
	.size	_Z15axpb_simd_m_mnmILi4EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi4EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi4EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC19:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 4]"
	.section	.text._Z11axpb_simd_cILi4EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi4EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi4EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi4EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi4EEl14axpb_options_tfPff:
.LFB5725:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$64, 56(%rbp)
	jne	.L116
	cmpq	$1, 32(%rbp)
	jne	.L117
	movq	64(%rbp), %rdx
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$4>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L112
	vbroadcastss	%xmm0, %zmm2
	vmovups	(%rdi), %zmm6
	vmovups	64(%rdi), %zmm5
	vmovups	128(%rdi), %zmm4
	vmovups	192(%rdi), %zmm3
	vbroadcastss	%xmm1, %zmm0
	xorl	%eax, %eax
	jmp	.L113
	.p2align 4,,10
	.p2align 3
.L114:
	vmovaps	%zmm1, %zmm3
.L113:
	vfmadd132ps	%zmm2, %zmm0, %zmm3
	incq	%rax
	vfmadd132ps	%zmm2, %zmm0, %zmm6
	vfmadd132ps	%zmm2, %zmm0, %zmm5
	vfmadd132ps	%zmm2, %zmm0, %zmm4
	vmovaps	%zmm3, %zmm1
	cmpq	%rax, %rdx
	jne	.L114
	vmovups	%zmm6, (%rdi)
	vmovups	%zmm5, 64(%rdi)
	vmovups	%zmm4, 128(%rdi)
	vmovups	%zmm3, 192(%rdi)
	vzeroupper
.L112:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$4>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L116:
	.cfi_restore_state
	leaq	.LC19(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L117:
	leaq	.LC19(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5725:
	.size	_Z11axpb_simd_cILi4EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi4EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi5EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC20:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 5]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi5EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi5EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi5EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi5EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi5EEl14axpb_options_tfPff:
.LFB5730:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC20(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5730:
	.size	_Z24axpb_simd_parallel_m_mnmILi5EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi5EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi5EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC21:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 5]"
	.section	.text._Z15axpb_simd_m_mnmILi5EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi5EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi5EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi5EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi5EEl14axpb_options_tfPff:
.LFB5729:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$7378697629483820647, %rdx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %r9
	movq	64(%rbp), %rcx
	movq	%r9, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%r9, %rdx
	sarq	$5, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	leaq	(%rax,%rax,4), %rax
	salq	$4, %rax
	cmpq	%rax, %r9
	jne	.L132
	cmpq	$1, 32(%rbp)
	jne	.L133
	testq	%r9, %r9
	leaq	15(%r9), %r8
	cmovns	%r9, %r8
	vmovaps	%xmm0, %xmm2
	vmovaps	%xmm1, %xmm0
	sarq	$4, %r8
	movq	%rdi, %rdx
	xorl	%esi, %esi
	vbroadcastss	%xmm2, %zmm1
	vbroadcastss	%xmm0, %zmm0
	cmpq	$15, %r9
	jle	.L129
	.p2align 4,,10
	.p2align 3
.L127:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$5>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rcx, %rcx
	jle	.L124
	vmovups	(%rdx), %zmm6
	vmovups	64(%rdx), %zmm5
	vmovups	128(%rdx), %zmm4
	vmovups	192(%rdx), %zmm3
	vmovups	256(%rdx), %zmm2
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L125:
	incq	%rax
	vfmadd132ps	%zmm1, %zmm0, %zmm6
	vfmadd132ps	%zmm1, %zmm0, %zmm5
	vfmadd132ps	%zmm1, %zmm0, %zmm4
	vfmadd132ps	%zmm1, %zmm0, %zmm3
	vfmadd132ps	%zmm1, %zmm0, %zmm2
	cmpq	%rax, %rcx
	jne	.L125
	vmovups	%zmm6, (%rdx)
	vmovups	%zmm5, 64(%rdx)
	vmovups	%zmm4, 128(%rdx)
	vmovups	%zmm3, 192(%rdx)
	vmovups	%zmm2, 256(%rdx)
.L124:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$5>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$5, %rsi
	addq	$320, %rdx
	cmpq	%r8, %rsi
	jl	.L127
.L129:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L132:
	.cfi_restore_state
	leaq	.LC21(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L133:
	leaq	.LC21(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5729:
	.size	_Z15axpb_simd_m_mnmILi5EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi5EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi5EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC22:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 5]"
	.section	.text._Z11axpb_simd_cILi5EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi5EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi5EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi5EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi5EEl14axpb_options_tfPff:
.LFB5728:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$80, 56(%rbp)
	jne	.L141
	cmpq	$1, 32(%rbp)
	jne	.L142
	movq	64(%rbp), %rdx
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$5>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L137
	vbroadcastss	%xmm0, %zmm2
	vmovups	(%rdi), %zmm7
	vmovups	64(%rdi), %zmm6
	vmovups	128(%rdi), %zmm5
	vmovups	192(%rdi), %zmm4
	vmovups	256(%rdi), %zmm3
	vbroadcastss	%xmm1, %zmm0
	xorl	%eax, %eax
	jmp	.L138
	.p2align 4,,10
	.p2align 3
.L139:
	vmovaps	%zmm1, %zmm3
.L138:
	vfmadd132ps	%zmm2, %zmm0, %zmm3
	incq	%rax
	vfmadd132ps	%zmm2, %zmm0, %zmm7
	vfmadd132ps	%zmm2, %zmm0, %zmm6
	vfmadd132ps	%zmm2, %zmm0, %zmm5
	vfmadd132ps	%zmm2, %zmm0, %zmm4
	vmovaps	%zmm3, %zmm1
	cmpq	%rax, %rdx
	jne	.L139
	vmovups	%zmm7, (%rdi)
	vmovups	%zmm6, 64(%rdi)
	vmovups	%zmm5, 128(%rdi)
	vmovups	%zmm4, 192(%rdi)
	vmovups	%zmm3, 256(%rdi)
	vzeroupper
.L137:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$5>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L141:
	.cfi_restore_state
	leaq	.LC22(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L142:
	leaq	.LC22(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5728:
	.size	_Z11axpb_simd_cILi5EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi5EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi6EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC23:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 6]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi6EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi6EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi6EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi6EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi6EEl14axpb_options_tfPff:
.LFB5733:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC23(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5733:
	.size	_Z24axpb_simd_parallel_m_mnmILi6EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi6EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi6EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC24:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 6]"
	.section	.text._Z15axpb_simd_m_mnmILi6EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi6EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi6EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi6EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi6EEl14axpb_options_tfPff:
.LFB5732:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$3074457345618258603, %rdx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %r9
	movq	64(%rbp), %rcx
	movq	%r9, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%r9, %rdx
	sarq	$4, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	leaq	(%rax,%rax,2), %rax
	salq	$5, %rax
	cmpq	%rax, %r9
	jne	.L157
	cmpq	$1, 32(%rbp)
	jne	.L158
	testq	%r9, %r9
	leaq	15(%r9), %r8
	cmovns	%r9, %r8
	vmovaps	%xmm0, %xmm2
	vmovaps	%xmm1, %xmm0
	sarq	$4, %r8
	movq	%rdi, %rdx
	xorl	%esi, %esi
	vbroadcastss	%xmm2, %zmm1
	vbroadcastss	%xmm0, %zmm0
	cmpq	$15, %r9
	jle	.L154
	.p2align 4,,10
	.p2align 3
.L152:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$6>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rcx, %rcx
	jle	.L149
	vmovups	(%rdx), %zmm7
	vmovups	64(%rdx), %zmm6
	vmovups	128(%rdx), %zmm5
	vmovups	192(%rdx), %zmm4
	vmovups	256(%rdx), %zmm3
	vmovups	320(%rdx), %zmm2
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L150:
	incq	%rax
	vfmadd132ps	%zmm1, %zmm0, %zmm7
	vfmadd132ps	%zmm1, %zmm0, %zmm6
	vfmadd132ps	%zmm1, %zmm0, %zmm5
	vfmadd132ps	%zmm1, %zmm0, %zmm4
	vfmadd132ps	%zmm1, %zmm0, %zmm3
	vfmadd132ps	%zmm1, %zmm0, %zmm2
	cmpq	%rax, %rcx
	jne	.L150
	vmovups	%zmm7, (%rdx)
	vmovups	%zmm6, 64(%rdx)
	vmovups	%zmm5, 128(%rdx)
	vmovups	%zmm4, 192(%rdx)
	vmovups	%zmm3, 256(%rdx)
	vmovups	%zmm2, 320(%rdx)
.L149:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$6>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$6, %rsi
	addq	$384, %rdx
	cmpq	%r8, %rsi
	jl	.L152
.L154:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L157:
	.cfi_restore_state
	leaq	.LC24(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L158:
	leaq	.LC24(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5732:
	.size	_Z15axpb_simd_m_mnmILi6EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi6EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi6EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC25:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 6]"
	.section	.text._Z11axpb_simd_cILi6EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi6EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi6EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi6EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi6EEl14axpb_options_tfPff:
.LFB5731:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$96, 56(%rbp)
	jne	.L166
	cmpq	$1, 32(%rbp)
	jne	.L167
	movq	64(%rbp), %rdx
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$6>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L162
	vbroadcastss	%xmm0, %zmm2
	vmovups	(%rdi), %zmm8
	vmovups	64(%rdi), %zmm7
	vmovups	128(%rdi), %zmm6
	vmovups	192(%rdi), %zmm5
	vmovups	256(%rdi), %zmm4
	vmovups	320(%rdi), %zmm3
	vbroadcastss	%xmm1, %zmm0
	xorl	%eax, %eax
	jmp	.L163
	.p2align 4,,10
	.p2align 3
.L164:
	vmovaps	%zmm1, %zmm3
.L163:
	vfmadd132ps	%zmm2, %zmm0, %zmm3
	incq	%rax
	vfmadd132ps	%zmm2, %zmm0, %zmm8
	vfmadd132ps	%zmm2, %zmm0, %zmm7
	vfmadd132ps	%zmm2, %zmm0, %zmm6
	vfmadd132ps	%zmm2, %zmm0, %zmm5
	vfmadd132ps	%zmm2, %zmm0, %zmm4
	vmovaps	%zmm3, %zmm1
	cmpq	%rax, %rdx
	jne	.L164
	vmovups	%zmm8, (%rdi)
	vmovups	%zmm7, 64(%rdi)
	vmovups	%zmm6, 128(%rdi)
	vmovups	%zmm5, 192(%rdi)
	vmovups	%zmm4, 256(%rdi)
	vmovups	%zmm3, 320(%rdi)
	vzeroupper
.L162:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$6>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L166:
	.cfi_restore_state
	leaq	.LC25(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L167:
	leaq	.LC25(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5731:
	.size	_Z11axpb_simd_cILi6EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi6EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi7EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC26:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 7]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi7EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi7EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi7EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi7EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi7EEl14axpb_options_tfPff:
.LFB5736:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC26(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5736:
	.size	_Z24axpb_simd_parallel_m_mnmILi7EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi7EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi7EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC27:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 7]"
	.section	.text._Z15axpb_simd_m_mnmILi7EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi7EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi7EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi7EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi7EEl14axpb_options_tfPff:
.LFB5735:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$5270498306774157605, %rdx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %r8
	movq	64(%rbp), %rsi
	movq	%r8, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%r8, %rdx
	sarq	$5, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$112, %rax, %rax
	cmpq	%rax, %r8
	jne	.L182
	cmpq	$1, 32(%rbp)
	jne	.L183
	movq	%rdi, %rcx
	testq	%r8, %r8
	leaq	15(%r8), %rdi
	cmovns	%r8, %rdi
	vmovaps	%xmm0, %xmm2
	vmovaps	%xmm1, %xmm0
	movq	%rcx, %rdx
	sarq	$4, %rdi
	xorl	%ecx, %ecx
	vbroadcastss	%xmm2, %zmm1
	vbroadcastss	%xmm0, %zmm0
	cmpq	$15, %r8
	jle	.L179
	.p2align 4,,10
	.p2align 3
.L177:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$7>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L174
	vmovups	(%rdx), %zmm8
	vmovups	64(%rdx), %zmm7
	vmovups	128(%rdx), %zmm6
	vmovups	192(%rdx), %zmm5
	vmovups	256(%rdx), %zmm4
	vmovups	320(%rdx), %zmm3
	vmovups	384(%rdx), %zmm2
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L175:
	incq	%rax
	vfmadd132ps	%zmm1, %zmm0, %zmm8
	vfmadd132ps	%zmm1, %zmm0, %zmm7
	vfmadd132ps	%zmm1, %zmm0, %zmm6
	vfmadd132ps	%zmm1, %zmm0, %zmm5
	vfmadd132ps	%zmm1, %zmm0, %zmm4
	vfmadd132ps	%zmm1, %zmm0, %zmm3
	vfmadd132ps	%zmm1, %zmm0, %zmm2
	cmpq	%rax, %rsi
	jne	.L175
	vmovups	%zmm8, (%rdx)
	vmovups	%zmm7, 64(%rdx)
	vmovups	%zmm6, 128(%rdx)
	vmovups	%zmm5, 192(%rdx)
	vmovups	%zmm4, 256(%rdx)
	vmovups	%zmm3, 320(%rdx)
	vmovups	%zmm2, 384(%rdx)
.L174:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$7>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$7, %rcx
	addq	$448, %rdx
	cmpq	%rdi, %rcx
	jl	.L177
.L179:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L182:
	.cfi_restore_state
	leaq	.LC27(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L183:
	leaq	.LC27(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5735:
	.size	_Z15axpb_simd_m_mnmILi7EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi7EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi7EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC28:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 7]"
	.section	.text._Z11axpb_simd_cILi7EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi7EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi7EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi7EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi7EEl14axpb_options_tfPff:
.LFB5734:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$112, 56(%rbp)
	jne	.L191
	cmpq	$1, 32(%rbp)
	jne	.L192
	movq	64(%rbp), %rdx
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$7>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L187
	vbroadcastss	%xmm0, %zmm2
	vmovups	(%rdi), %zmm9
	vmovups	64(%rdi), %zmm8
	vmovups	128(%rdi), %zmm7
	vmovups	192(%rdi), %zmm6
	vmovups	256(%rdi), %zmm5
	vmovups	320(%rdi), %zmm4
	vmovups	384(%rdi), %zmm3
	vbroadcastss	%xmm1, %zmm0
	xorl	%eax, %eax
	jmp	.L188
	.p2align 4,,10
	.p2align 3
.L189:
	vmovaps	%zmm1, %zmm3
.L188:
	vfmadd132ps	%zmm2, %zmm0, %zmm3
	incq	%rax
	vfmadd132ps	%zmm2, %zmm0, %zmm9
	vfmadd132ps	%zmm2, %zmm0, %zmm8
	vfmadd132ps	%zmm2, %zmm0, %zmm7
	vfmadd132ps	%zmm2, %zmm0, %zmm6
	vfmadd132ps	%zmm2, %zmm0, %zmm5
	vfmadd132ps	%zmm2, %zmm0, %zmm4
	vmovaps	%zmm3, %zmm1
	cmpq	%rax, %rdx
	jne	.L189
	vmovups	%zmm9, (%rdi)
	vmovups	%zmm8, 64(%rdi)
	vmovups	%zmm7, 128(%rdi)
	vmovups	%zmm6, 192(%rdi)
	vmovups	%zmm5, 256(%rdi)
	vmovups	%zmm4, 320(%rdi)
	vmovups	%zmm3, 384(%rdi)
	vzeroupper
.L187:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$7>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L191:
	.cfi_restore_state
	leaq	.LC28(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L192:
	leaq	.LC28(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5734:
	.size	_Z11axpb_simd_cILi7EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi7EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi8EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC29:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 8]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi8EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi8EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi8EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi8EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi8EEl14axpb_options_tfPff:
.LFB5739:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC29(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5739:
	.size	_Z24axpb_simd_parallel_m_mnmILi8EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi8EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi8EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC30:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 8]"
	.section	.text._Z15axpb_simd_m_mnmILi8EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi8EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi8EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi8EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi8EEl14axpb_options_tfPff:
.LFB5738:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rax
	movq	64(%rbp), %rcx
	andq	$-64, %rsp
	testb	$127, %al
	jne	.L207
	cmpq	$1, 32(%rbp)
	jne	.L208
	testq	%rax, %rax
	movq	%rdi, %rdx
	leaq	15(%rax), %rdi
	cmovns	%rax, %rdi
	vmovaps	%xmm0, %xmm2
	vmovaps	%xmm1, %xmm0
	sarq	$4, %rdi
	xorl	%esi, %esi
	vbroadcastss	%xmm2, %zmm1
	vbroadcastss	%xmm0, %zmm0
	cmpq	$15, %rax
	jle	.L204
	.p2align 4,,10
	.p2align 3
.L202:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$8>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rcx, %rcx
	jle	.L199
	vmovups	(%rdx), %zmm9
	vmovups	64(%rdx), %zmm8
	vmovups	128(%rdx), %zmm7
	vmovups	192(%rdx), %zmm6
	vmovups	256(%rdx), %zmm5
	vmovups	320(%rdx), %zmm4
	vmovups	384(%rdx), %zmm3
	vmovups	448(%rdx), %zmm2
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L200:
	incq	%rax
	vfmadd132ps	%zmm1, %zmm0, %zmm9
	vfmadd132ps	%zmm1, %zmm0, %zmm8
	vfmadd132ps	%zmm1, %zmm0, %zmm7
	vfmadd132ps	%zmm1, %zmm0, %zmm6
	vfmadd132ps	%zmm1, %zmm0, %zmm5
	vfmadd132ps	%zmm1, %zmm0, %zmm4
	vfmadd132ps	%zmm1, %zmm0, %zmm3
	vfmadd132ps	%zmm1, %zmm0, %zmm2
	cmpq	%rax, %rcx
	jne	.L200
	vmovups	%zmm9, (%rdx)
	vmovups	%zmm8, 64(%rdx)
	vmovups	%zmm7, 128(%rdx)
	vmovups	%zmm6, 192(%rdx)
	vmovups	%zmm5, 256(%rdx)
	vmovups	%zmm4, 320(%rdx)
	vmovups	%zmm3, 384(%rdx)
	vmovups	%zmm2, 448(%rdx)
.L199:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$8>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$8, %rsi
	addq	$512, %rdx
	cmpq	%rdi, %rsi
	jl	.L202
.L204:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L207:
	.cfi_restore_state
	leaq	.LC30(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L208:
	leaq	.LC30(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5738:
	.size	_Z15axpb_simd_m_mnmILi8EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi8EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi8EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC31:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 8]"
	.section	.text._Z11axpb_simd_cILi8EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi8EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi8EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi8EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi8EEl14axpb_options_tfPff:
.LFB5737:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$128, 56(%rbp)
	jne	.L216
	cmpq	$1, 32(%rbp)
	jne	.L217
	movq	64(%rbp), %rdx
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$8>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L212
	vbroadcastss	%xmm0, %zmm2
	vmovups	(%rdi), %zmm10
	vmovups	64(%rdi), %zmm9
	vmovups	128(%rdi), %zmm8
	vmovups	192(%rdi), %zmm7
	vmovups	256(%rdi), %zmm6
	vmovups	320(%rdi), %zmm5
	vmovups	384(%rdi), %zmm4
	vmovups	448(%rdi), %zmm3
	vbroadcastss	%xmm1, %zmm0
	xorl	%eax, %eax
	jmp	.L213
	.p2align 4,,10
	.p2align 3
.L214:
	vmovaps	%zmm1, %zmm3
.L213:
	vfmadd132ps	%zmm2, %zmm0, %zmm3
	incq	%rax
	vfmadd132ps	%zmm2, %zmm0, %zmm10
	vfmadd132ps	%zmm2, %zmm0, %zmm9
	vfmadd132ps	%zmm2, %zmm0, %zmm8
	vfmadd132ps	%zmm2, %zmm0, %zmm7
	vfmadd132ps	%zmm2, %zmm0, %zmm6
	vfmadd132ps	%zmm2, %zmm0, %zmm5
	vfmadd132ps	%zmm2, %zmm0, %zmm4
	vmovaps	%zmm3, %zmm1
	cmpq	%rax, %rdx
	jne	.L214
	vmovups	%zmm10, (%rdi)
	vmovups	%zmm9, 64(%rdi)
	vmovups	%zmm8, 128(%rdi)
	vmovups	%zmm7, 192(%rdi)
	vmovups	%zmm6, 256(%rdi)
	vmovups	%zmm5, 320(%rdi)
	vmovups	%zmm4, 384(%rdi)
	vmovups	%zmm3, 448(%rdi)
	vzeroupper
.L212:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$8>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L216:
	.cfi_restore_state
	leaq	.LC31(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L217:
	leaq	.LC31(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5737:
	.size	_Z11axpb_simd_cILi8EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi8EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi9EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC32:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 9]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi9EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi9EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi9EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi9EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi9EEl14axpb_options_tfPff:
.LFB5742:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC32(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5742:
	.size	_Z24axpb_simd_parallel_m_mnmILi9EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi9EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi9EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC33:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 9]"
	.section	.text._Z15axpb_simd_m_mnmILi9EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi9EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi9EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi9EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi9EEl14axpb_options_tfPff:
.LFB5741:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$1024819115206086201, %rdx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %r8
	movq	64(%rbp), %rsi
	movq	%r8, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%r8, %rdx
	sarq	$3, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	leaq	(%rax,%rax,8), %rax
	salq	$4, %rax
	cmpq	%rax, %r8
	jne	.L232
	cmpq	$1, 32(%rbp)
	jne	.L233
	movq	%rdi, %rcx
	testq	%r8, %r8
	leaq	15(%r8), %rdi
	cmovns	%r8, %rdi
	vmovaps	%xmm0, %xmm2
	vmovaps	%xmm1, %xmm0
	movq	%rcx, %rdx
	sarq	$4, %rdi
	xorl	%ecx, %ecx
	vbroadcastss	%xmm2, %zmm1
	vbroadcastss	%xmm0, %zmm0
	cmpq	$15, %r8
	jle	.L229
	.p2align 4,,10
	.p2align 3
.L227:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$9>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L224
	vmovups	(%rdx), %zmm10
	vmovups	64(%rdx), %zmm9
	vmovups	128(%rdx), %zmm8
	vmovups	192(%rdx), %zmm7
	vmovups	256(%rdx), %zmm6
	vmovups	320(%rdx), %zmm5
	vmovups	384(%rdx), %zmm4
	vmovups	448(%rdx), %zmm3
	vmovups	512(%rdx), %zmm2
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L225:
	incq	%rax
	vfmadd132ps	%zmm1, %zmm0, %zmm10
	vfmadd132ps	%zmm1, %zmm0, %zmm9
	vfmadd132ps	%zmm1, %zmm0, %zmm8
	vfmadd132ps	%zmm1, %zmm0, %zmm7
	vfmadd132ps	%zmm1, %zmm0, %zmm6
	vfmadd132ps	%zmm1, %zmm0, %zmm5
	vfmadd132ps	%zmm1, %zmm0, %zmm4
	vfmadd132ps	%zmm1, %zmm0, %zmm3
	vfmadd132ps	%zmm1, %zmm0, %zmm2
	cmpq	%rax, %rsi
	jne	.L225
	vmovups	%zmm10, (%rdx)
	vmovups	%zmm9, 64(%rdx)
	vmovups	%zmm8, 128(%rdx)
	vmovups	%zmm7, 192(%rdx)
	vmovups	%zmm6, 256(%rdx)
	vmovups	%zmm5, 320(%rdx)
	vmovups	%zmm4, 384(%rdx)
	vmovups	%zmm3, 448(%rdx)
	vmovups	%zmm2, 512(%rdx)
.L224:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$9>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$9, %rcx
	addq	$576, %rdx
	cmpq	%rdi, %rcx
	jl	.L227
.L229:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L232:
	.cfi_restore_state
	leaq	.LC33(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L233:
	leaq	.LC33(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5741:
	.size	_Z15axpb_simd_m_mnmILi9EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi9EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi9EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC34:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 9]"
	.section	.text._Z11axpb_simd_cILi9EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi9EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi9EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi9EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi9EEl14axpb_options_tfPff:
.LFB5740:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$144, 56(%rbp)
	jne	.L241
	cmpq	$1, 32(%rbp)
	jne	.L242
	movq	64(%rbp), %rdx
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$9>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L237
	vbroadcastss	%xmm0, %zmm2
	vmovups	(%rdi), %zmm11
	vmovups	64(%rdi), %zmm10
	vmovups	128(%rdi), %zmm9
	vmovups	192(%rdi), %zmm8
	vmovups	256(%rdi), %zmm7
	vmovups	320(%rdi), %zmm6
	vmovups	384(%rdi), %zmm5
	vmovups	448(%rdi), %zmm4
	vmovups	512(%rdi), %zmm3
	vbroadcastss	%xmm1, %zmm0
	xorl	%eax, %eax
	jmp	.L238
	.p2align 4,,10
	.p2align 3
.L239:
	vmovaps	%zmm1, %zmm3
.L238:
	vfmadd132ps	%zmm2, %zmm0, %zmm3
	incq	%rax
	vfmadd132ps	%zmm2, %zmm0, %zmm11
	vfmadd132ps	%zmm2, %zmm0, %zmm10
	vfmadd132ps	%zmm2, %zmm0, %zmm9
	vfmadd132ps	%zmm2, %zmm0, %zmm8
	vfmadd132ps	%zmm2, %zmm0, %zmm7
	vfmadd132ps	%zmm2, %zmm0, %zmm6
	vfmadd132ps	%zmm2, %zmm0, %zmm5
	vfmadd132ps	%zmm2, %zmm0, %zmm4
	vmovaps	%zmm3, %zmm1
	cmpq	%rax, %rdx
	jne	.L239
	vmovups	%zmm11, (%rdi)
	vmovups	%zmm10, 64(%rdi)
	vmovups	%zmm9, 128(%rdi)
	vmovups	%zmm8, 192(%rdi)
	vmovups	%zmm7, 256(%rdi)
	vmovups	%zmm6, 320(%rdi)
	vmovups	%zmm5, 384(%rdi)
	vmovups	%zmm4, 448(%rdi)
	vmovups	%zmm3, 512(%rdi)
	vzeroupper
.L237:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$9>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L241:
	.cfi_restore_state
	leaq	.LC34(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L242:
	leaq	.LC34(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5740:
	.size	_Z11axpb_simd_cILi9EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi9EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi10EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC35:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 10]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi10EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi10EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi10EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi10EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi10EEl14axpb_options_tfPff:
.LFB5745:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC35(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5745:
	.size	_Z24axpb_simd_parallel_m_mnmILi10EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi10EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi10EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC36:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 10]"
	.section	.text._Z15axpb_simd_m_mnmILi10EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi10EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi10EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi10EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi10EEl14axpb_options_tfPff:
.LFB5744:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$7378697629483820647, %rdx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %r8
	movq	64(%rbp), %rsi
	movq	%r8, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%r8, %rdx
	sarq	$6, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	leaq	(%rax,%rax,4), %rax
	salq	$5, %rax
	cmpq	%rax, %r8
	jne	.L257
	cmpq	$1, 32(%rbp)
	jne	.L258
	movq	%rdi, %rcx
	testq	%r8, %r8
	leaq	15(%r8), %rdi
	cmovns	%r8, %rdi
	vmovaps	%xmm0, %xmm2
	vmovaps	%xmm1, %xmm0
	movq	%rcx, %rdx
	sarq	$4, %rdi
	xorl	%ecx, %ecx
	vbroadcastss	%xmm2, %zmm1
	vbroadcastss	%xmm0, %zmm0
	cmpq	$15, %r8
	jle	.L254
	.p2align 4,,10
	.p2align 3
.L252:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$10>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L249
	vmovups	(%rdx), %zmm11
	vmovups	64(%rdx), %zmm10
	vmovups	128(%rdx), %zmm9
	vmovups	192(%rdx), %zmm8
	vmovups	256(%rdx), %zmm7
	vmovups	320(%rdx), %zmm6
	vmovups	384(%rdx), %zmm5
	vmovups	448(%rdx), %zmm4
	vmovups	512(%rdx), %zmm3
	vmovups	576(%rdx), %zmm2
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L250:
	incq	%rax
	vfmadd132ps	%zmm1, %zmm0, %zmm11
	vfmadd132ps	%zmm1, %zmm0, %zmm10
	vfmadd132ps	%zmm1, %zmm0, %zmm9
	vfmadd132ps	%zmm1, %zmm0, %zmm8
	vfmadd132ps	%zmm1, %zmm0, %zmm7
	vfmadd132ps	%zmm1, %zmm0, %zmm6
	vfmadd132ps	%zmm1, %zmm0, %zmm5
	vfmadd132ps	%zmm1, %zmm0, %zmm4
	vfmadd132ps	%zmm1, %zmm0, %zmm3
	vfmadd132ps	%zmm1, %zmm0, %zmm2
	cmpq	%rax, %rsi
	jne	.L250
	vmovups	%zmm11, (%rdx)
	vmovups	%zmm10, 64(%rdx)
	vmovups	%zmm9, 128(%rdx)
	vmovups	%zmm8, 192(%rdx)
	vmovups	%zmm7, 256(%rdx)
	vmovups	%zmm6, 320(%rdx)
	vmovups	%zmm5, 384(%rdx)
	vmovups	%zmm4, 448(%rdx)
	vmovups	%zmm3, 512(%rdx)
	vmovups	%zmm2, 576(%rdx)
.L249:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$10>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$10, %rcx
	addq	$640, %rdx
	cmpq	%rdi, %rcx
	jl	.L252
.L254:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L257:
	.cfi_restore_state
	leaq	.LC36(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L258:
	leaq	.LC36(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5744:
	.size	_Z15axpb_simd_m_mnmILi10EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi10EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi10EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC37:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 10]"
	.section	.text._Z11axpb_simd_cILi10EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi10EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi10EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi10EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi10EEl14axpb_options_tfPff:
.LFB5743:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$160, 56(%rbp)
	jne	.L266
	cmpq	$1, 32(%rbp)
	jne	.L267
	movq	64(%rbp), %rdx
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$10>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L262
	vbroadcastss	%xmm0, %zmm2
	vmovups	(%rdi), %zmm12
	vmovups	64(%rdi), %zmm11
	vmovups	128(%rdi), %zmm10
	vmovups	192(%rdi), %zmm9
	vmovups	256(%rdi), %zmm8
	vmovups	320(%rdi), %zmm7
	vmovups	384(%rdi), %zmm6
	vmovups	448(%rdi), %zmm5
	vmovups	512(%rdi), %zmm4
	vmovups	576(%rdi), %zmm3
	vbroadcastss	%xmm1, %zmm0
	xorl	%eax, %eax
	jmp	.L263
	.p2align 4,,10
	.p2align 3
.L264:
	vmovaps	%zmm1, %zmm3
.L263:
	vfmadd132ps	%zmm2, %zmm0, %zmm3
	incq	%rax
	vfmadd132ps	%zmm2, %zmm0, %zmm12
	vfmadd132ps	%zmm2, %zmm0, %zmm11
	vfmadd132ps	%zmm2, %zmm0, %zmm10
	vfmadd132ps	%zmm2, %zmm0, %zmm9
	vfmadd132ps	%zmm2, %zmm0, %zmm8
	vfmadd132ps	%zmm2, %zmm0, %zmm7
	vfmadd132ps	%zmm2, %zmm0, %zmm6
	vfmadd132ps	%zmm2, %zmm0, %zmm5
	vfmadd132ps	%zmm2, %zmm0, %zmm4
	vmovaps	%zmm3, %zmm1
	cmpq	%rax, %rdx
	jne	.L264
	vmovups	%zmm12, (%rdi)
	vmovups	%zmm11, 64(%rdi)
	vmovups	%zmm10, 128(%rdi)
	vmovups	%zmm9, 192(%rdi)
	vmovups	%zmm8, 256(%rdi)
	vmovups	%zmm7, 320(%rdi)
	vmovups	%zmm6, 384(%rdi)
	vmovups	%zmm5, 448(%rdi)
	vmovups	%zmm4, 512(%rdi)
	vmovups	%zmm3, 576(%rdi)
	vzeroupper
.L262:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$10>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L266:
	.cfi_restore_state
	leaq	.LC37(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L267:
	leaq	.LC37(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5743:
	.size	_Z11axpb_simd_cILi10EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi10EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi11EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC38:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 11]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi11EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi11EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi11EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi11EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi11EEl14axpb_options_tfPff:
.LFB5748:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC38(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5748:
	.size	_Z24axpb_simd_parallel_m_mnmILi11EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi11EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi11EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC39:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 11]"
	.section	.text._Z15axpb_simd_m_mnmILi11EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi11EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi11EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi11EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi11EEl14axpb_options_tfPff:
.LFB5747:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$3353953467947191203, %rdx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %r8
	movq	64(%rbp), %rsi
	movq	%r8, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%r8, %rdx
	sarq	$5, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$176, %rax, %rax
	cmpq	%rax, %r8
	jne	.L282
	cmpq	$1, 32(%rbp)
	jne	.L283
	movq	%rdi, %rcx
	testq	%r8, %r8
	leaq	15(%r8), %rdi
	cmovns	%r8, %rdi
	vmovaps	%xmm0, %xmm2
	vmovaps	%xmm1, %xmm0
	movq	%rcx, %rdx
	sarq	$4, %rdi
	xorl	%ecx, %ecx
	vbroadcastss	%xmm2, %zmm1
	vbroadcastss	%xmm0, %zmm0
	cmpq	$15, %r8
	jle	.L279
	.p2align 4,,10
	.p2align 3
.L277:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$11>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L274
	vmovups	(%rdx), %zmm12
	vmovups	64(%rdx), %zmm11
	vmovups	128(%rdx), %zmm10
	vmovups	192(%rdx), %zmm9
	vmovups	256(%rdx), %zmm8
	vmovups	320(%rdx), %zmm7
	vmovups	384(%rdx), %zmm6
	vmovups	448(%rdx), %zmm5
	vmovups	512(%rdx), %zmm4
	vmovups	576(%rdx), %zmm3
	vmovups	640(%rdx), %zmm2
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L275:
	incq	%rax
	vfmadd132ps	%zmm1, %zmm0, %zmm12
	vfmadd132ps	%zmm1, %zmm0, %zmm11
	vfmadd132ps	%zmm1, %zmm0, %zmm10
	vfmadd132ps	%zmm1, %zmm0, %zmm9
	vfmadd132ps	%zmm1, %zmm0, %zmm8
	vfmadd132ps	%zmm1, %zmm0, %zmm7
	vfmadd132ps	%zmm1, %zmm0, %zmm6
	vfmadd132ps	%zmm1, %zmm0, %zmm5
	vfmadd132ps	%zmm1, %zmm0, %zmm4
	vfmadd132ps	%zmm1, %zmm0, %zmm3
	vfmadd132ps	%zmm1, %zmm0, %zmm2
	cmpq	%rax, %rsi
	jne	.L275
	vmovups	%zmm12, (%rdx)
	vmovups	%zmm11, 64(%rdx)
	vmovups	%zmm10, 128(%rdx)
	vmovups	%zmm9, 192(%rdx)
	vmovups	%zmm8, 256(%rdx)
	vmovups	%zmm7, 320(%rdx)
	vmovups	%zmm6, 384(%rdx)
	vmovups	%zmm5, 448(%rdx)
	vmovups	%zmm4, 512(%rdx)
	vmovups	%zmm3, 576(%rdx)
	vmovups	%zmm2, 640(%rdx)
.L274:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$11>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$11, %rcx
	addq	$704, %rdx
	cmpq	%rdi, %rcx
	jl	.L277
.L279:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L282:
	.cfi_restore_state
	leaq	.LC39(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L283:
	leaq	.LC39(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5747:
	.size	_Z15axpb_simd_m_mnmILi11EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi11EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi11EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC40:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 11]"
	.section	.text._Z11axpb_simd_cILi11EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi11EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi11EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi11EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi11EEl14axpb_options_tfPff:
.LFB5746:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$176, 56(%rbp)
	jne	.L291
	cmpq	$1, 32(%rbp)
	jne	.L292
	movq	64(%rbp), %rdx
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$11>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L287
	vbroadcastss	%xmm0, %zmm2
	vmovups	(%rdi), %zmm13
	vmovups	64(%rdi), %zmm12
	vmovups	128(%rdi), %zmm11
	vmovups	192(%rdi), %zmm10
	vmovups	256(%rdi), %zmm9
	vmovups	320(%rdi), %zmm8
	vmovups	384(%rdi), %zmm7
	vmovups	448(%rdi), %zmm6
	vmovups	512(%rdi), %zmm5
	vmovups	576(%rdi), %zmm4
	vmovups	640(%rdi), %zmm3
	vbroadcastss	%xmm1, %zmm0
	xorl	%eax, %eax
	jmp	.L288
	.p2align 4,,10
	.p2align 3
.L289:
	vmovaps	%zmm1, %zmm3
.L288:
	vfmadd132ps	%zmm2, %zmm0, %zmm3
	incq	%rax
	vfmadd132ps	%zmm2, %zmm0, %zmm13
	vfmadd132ps	%zmm2, %zmm0, %zmm12
	vfmadd132ps	%zmm2, %zmm0, %zmm11
	vfmadd132ps	%zmm2, %zmm0, %zmm10
	vfmadd132ps	%zmm2, %zmm0, %zmm9
	vfmadd132ps	%zmm2, %zmm0, %zmm8
	vfmadd132ps	%zmm2, %zmm0, %zmm7
	vfmadd132ps	%zmm2, %zmm0, %zmm6
	vfmadd132ps	%zmm2, %zmm0, %zmm5
	vfmadd132ps	%zmm2, %zmm0, %zmm4
	vmovaps	%zmm3, %zmm1
	cmpq	%rax, %rdx
	jne	.L289
	vmovups	%zmm13, (%rdi)
	vmovups	%zmm12, 64(%rdi)
	vmovups	%zmm11, 128(%rdi)
	vmovups	%zmm10, 192(%rdi)
	vmovups	%zmm9, 256(%rdi)
	vmovups	%zmm8, 320(%rdi)
	vmovups	%zmm7, 384(%rdi)
	vmovups	%zmm6, 448(%rdi)
	vmovups	%zmm5, 512(%rdi)
	vmovups	%zmm4, 576(%rdi)
	vmovups	%zmm3, 640(%rdi)
	vzeroupper
.L287:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$11>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L291:
	.cfi_restore_state
	leaq	.LC40(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L292:
	leaq	.LC40(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5746:
	.size	_Z11axpb_simd_cILi11EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi11EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi12EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC41:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 12]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi12EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi12EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi12EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi12EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi12EEl14axpb_options_tfPff:
.LFB5751:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC41(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5751:
	.size	_Z24axpb_simd_parallel_m_mnmILi12EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi12EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi12EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC42:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 12]"
	.section	.text._Z15axpb_simd_m_mnmILi12EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi12EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi12EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi12EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi12EEl14axpb_options_tfPff:
.LFB5750:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$3074457345618258603, %rdx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %r8
	movq	64(%rbp), %rsi
	movq	%r8, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%r8, %rdx
	sarq	$5, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	leaq	(%rax,%rax,2), %rax
	salq	$6, %rax
	cmpq	%rax, %r8
	jne	.L307
	cmpq	$1, 32(%rbp)
	jne	.L308
	movq	%rdi, %rcx
	testq	%r8, %r8
	leaq	15(%r8), %rdi
	cmovns	%r8, %rdi
	vmovaps	%xmm0, %xmm2
	vmovaps	%xmm1, %xmm0
	movq	%rcx, %rdx
	sarq	$4, %rdi
	xorl	%ecx, %ecx
	vbroadcastss	%xmm2, %zmm1
	vbroadcastss	%xmm0, %zmm0
	cmpq	$15, %r8
	jle	.L304
.L302:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$12>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L299
	vmovups	(%rdx), %zmm13
	vmovups	64(%rdx), %zmm12
	vmovups	128(%rdx), %zmm11
	vmovups	192(%rdx), %zmm10
	vmovups	256(%rdx), %zmm9
	vmovups	320(%rdx), %zmm8
	vmovups	384(%rdx), %zmm7
	vmovups	448(%rdx), %zmm6
	vmovups	512(%rdx), %zmm5
	vmovups	576(%rdx), %zmm4
	vmovups	640(%rdx), %zmm3
	vmovups	704(%rdx), %zmm2
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L300:
	incq	%rax
	vfmadd132ps	%zmm1, %zmm0, %zmm13
	vfmadd132ps	%zmm1, %zmm0, %zmm12
	vfmadd132ps	%zmm1, %zmm0, %zmm11
	vfmadd132ps	%zmm1, %zmm0, %zmm10
	vfmadd132ps	%zmm1, %zmm0, %zmm9
	vfmadd132ps	%zmm1, %zmm0, %zmm8
	vfmadd132ps	%zmm1, %zmm0, %zmm7
	vfmadd132ps	%zmm1, %zmm0, %zmm6
	vfmadd132ps	%zmm1, %zmm0, %zmm5
	vfmadd132ps	%zmm1, %zmm0, %zmm4
	vfmadd132ps	%zmm1, %zmm0, %zmm3
	vfmadd132ps	%zmm1, %zmm0, %zmm2
	cmpq	%rax, %rsi
	jne	.L300
	vmovups	%zmm13, (%rdx)
	vmovups	%zmm12, 64(%rdx)
	vmovups	%zmm11, 128(%rdx)
	vmovups	%zmm10, 192(%rdx)
	vmovups	%zmm9, 256(%rdx)
	vmovups	%zmm8, 320(%rdx)
	vmovups	%zmm7, 384(%rdx)
	vmovups	%zmm6, 448(%rdx)
	vmovups	%zmm5, 512(%rdx)
	vmovups	%zmm4, 576(%rdx)
	vmovups	%zmm3, 640(%rdx)
	vmovups	%zmm2, 704(%rdx)
.L299:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$12>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$12, %rcx
	addq	$768, %rdx
	cmpq	%rdi, %rcx
	jl	.L302
.L304:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L307:
	.cfi_restore_state
	leaq	.LC42(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L308:
	leaq	.LC42(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5750:
	.size	_Z15axpb_simd_m_mnmILi12EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi12EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi12EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC43:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 12]"
	.section	.text._Z11axpb_simd_cILi12EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi12EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi12EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi12EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi12EEl14axpb_options_tfPff:
.LFB5749:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$192, 56(%rbp)
	jne	.L316
	cmpq	$1, 32(%rbp)
	jne	.L317
	movq	64(%rbp), %rdx
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$12>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L312
	vbroadcastss	%xmm0, %zmm2
	vmovups	(%rdi), %zmm14
	vmovups	64(%rdi), %zmm13
	vmovups	128(%rdi), %zmm12
	vmovups	192(%rdi), %zmm11
	vmovups	256(%rdi), %zmm10
	vmovups	320(%rdi), %zmm9
	vmovups	384(%rdi), %zmm8
	vmovups	448(%rdi), %zmm7
	vmovups	512(%rdi), %zmm6
	vmovups	576(%rdi), %zmm5
	vmovups	640(%rdi), %zmm4
	vmovups	704(%rdi), %zmm3
	vbroadcastss	%xmm1, %zmm0
	xorl	%eax, %eax
	jmp	.L313
	.p2align 4,,10
	.p2align 3
.L314:
	vmovaps	%zmm1, %zmm3
.L313:
	vfmadd132ps	%zmm2, %zmm0, %zmm3
	incq	%rax
	vfmadd132ps	%zmm2, %zmm0, %zmm14
	vfmadd132ps	%zmm2, %zmm0, %zmm13
	vfmadd132ps	%zmm2, %zmm0, %zmm12
	vfmadd132ps	%zmm2, %zmm0, %zmm11
	vfmadd132ps	%zmm2, %zmm0, %zmm10
	vfmadd132ps	%zmm2, %zmm0, %zmm9
	vfmadd132ps	%zmm2, %zmm0, %zmm8
	vfmadd132ps	%zmm2, %zmm0, %zmm7
	vfmadd132ps	%zmm2, %zmm0, %zmm6
	vfmadd132ps	%zmm2, %zmm0, %zmm5
	vfmadd132ps	%zmm2, %zmm0, %zmm4
	vmovaps	%zmm3, %zmm1
	cmpq	%rax, %rdx
	jne	.L314
	vmovups	%zmm14, (%rdi)
	vmovups	%zmm13, 64(%rdi)
	vmovups	%zmm12, 128(%rdi)
	vmovups	%zmm11, 192(%rdi)
	vmovups	%zmm10, 256(%rdi)
	vmovups	%zmm9, 320(%rdi)
	vmovups	%zmm8, 384(%rdi)
	vmovups	%zmm7, 448(%rdi)
	vmovups	%zmm6, 512(%rdi)
	vmovups	%zmm5, 576(%rdi)
	vmovups	%zmm4, 640(%rdi)
	vmovups	%zmm3, 704(%rdi)
	vzeroupper
.L312:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$12>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L316:
	.cfi_restore_state
	leaq	.LC43(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L317:
	leaq	.LC43(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5749:
	.size	_Z11axpb_simd_cILi12EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi12EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi13EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC44:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 13]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi13EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi13EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi13EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi13EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi13EEl14axpb_options_tfPff:
.LFB5754:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC44(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5754:
	.size	_Z24axpb_simd_parallel_m_mnmILi13EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi13EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi13EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC45:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 13]"
	.section	.text._Z15axpb_simd_m_mnmILi13EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi13EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi13EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi13EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi13EEl14axpb_options_tfPff:
.LFB5753:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$5675921253449092805, %rdx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %r8
	movq	64(%rbp), %rsi
	movq	%r8, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%r8, %rdx
	sarq	$6, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$208, %rax, %rax
	cmpq	%rax, %r8
	jne	.L332
	cmpq	$1, 32(%rbp)
	jne	.L333
	movq	%rdi, %rcx
	testq	%r8, %r8
	leaq	15(%r8), %rdi
	cmovns	%r8, %rdi
	vmovaps	%xmm0, %xmm2
	vmovaps	%xmm1, %xmm0
	movq	%rcx, %rdx
	sarq	$4, %rdi
	xorl	%ecx, %ecx
	vbroadcastss	%xmm2, %zmm1
	vbroadcastss	%xmm0, %zmm0
	cmpq	$15, %r8
	jle	.L329
.L327:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$13>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L324
	vmovups	(%rdx), %zmm14
	vmovups	64(%rdx), %zmm13
	vmovups	128(%rdx), %zmm12
	vmovups	192(%rdx), %zmm11
	vmovups	256(%rdx), %zmm10
	vmovups	320(%rdx), %zmm9
	vmovups	384(%rdx), %zmm8
	vmovups	448(%rdx), %zmm7
	vmovups	512(%rdx), %zmm6
	vmovups	576(%rdx), %zmm5
	vmovups	640(%rdx), %zmm4
	vmovups	704(%rdx), %zmm3
	vmovups	768(%rdx), %zmm2
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L325:
	incq	%rax
	vfmadd132ps	%zmm1, %zmm0, %zmm14
	vfmadd132ps	%zmm1, %zmm0, %zmm13
	vfmadd132ps	%zmm1, %zmm0, %zmm12
	vfmadd132ps	%zmm1, %zmm0, %zmm11
	vfmadd132ps	%zmm1, %zmm0, %zmm10
	vfmadd132ps	%zmm1, %zmm0, %zmm9
	vfmadd132ps	%zmm1, %zmm0, %zmm8
	vfmadd132ps	%zmm1, %zmm0, %zmm7
	vfmadd132ps	%zmm1, %zmm0, %zmm6
	vfmadd132ps	%zmm1, %zmm0, %zmm5
	vfmadd132ps	%zmm1, %zmm0, %zmm4
	vfmadd132ps	%zmm1, %zmm0, %zmm3
	vfmadd132ps	%zmm1, %zmm0, %zmm2
	cmpq	%rax, %rsi
	jne	.L325
	vmovups	%zmm14, (%rdx)
	vmovups	%zmm13, 64(%rdx)
	vmovups	%zmm12, 128(%rdx)
	vmovups	%zmm11, 192(%rdx)
	vmovups	%zmm10, 256(%rdx)
	vmovups	%zmm9, 320(%rdx)
	vmovups	%zmm8, 384(%rdx)
	vmovups	%zmm7, 448(%rdx)
	vmovups	%zmm6, 512(%rdx)
	vmovups	%zmm5, 576(%rdx)
	vmovups	%zmm4, 640(%rdx)
	vmovups	%zmm3, 704(%rdx)
	vmovups	%zmm2, 768(%rdx)
.L324:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$13>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$13, %rcx
	addq	$832, %rdx
	cmpq	%rdi, %rcx
	jl	.L327
.L329:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L332:
	.cfi_restore_state
	leaq	.LC45(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L333:
	leaq	.LC45(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5753:
	.size	_Z15axpb_simd_m_mnmILi13EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi13EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi13EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC46:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 13]"
	.section	.text._Z11axpb_simd_cILi13EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi13EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi13EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi13EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi13EEl14axpb_options_tfPff:
.LFB5752:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$208, 56(%rbp)
	jne	.L341
	cmpq	$1, 32(%rbp)
	jne	.L342
	movq	64(%rbp), %rdx
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$13>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L337
	vbroadcastss	%xmm0, %zmm2
	vmovups	(%rdi), %zmm15
	vmovups	64(%rdi), %zmm14
	vmovups	128(%rdi), %zmm13
	vmovups	192(%rdi), %zmm12
	vmovups	256(%rdi), %zmm11
	vmovups	320(%rdi), %zmm10
	vmovups	384(%rdi), %zmm9
	vmovups	448(%rdi), %zmm8
	vmovups	512(%rdi), %zmm7
	vmovups	576(%rdi), %zmm6
	vmovups	640(%rdi), %zmm5
	vmovups	704(%rdi), %zmm4
	vmovups	768(%rdi), %zmm3
	vbroadcastss	%xmm1, %zmm0
	xorl	%eax, %eax
	jmp	.L338
	.p2align 4,,10
	.p2align 3
.L339:
	vmovaps	%zmm1, %zmm3
.L338:
	vfmadd132ps	%zmm2, %zmm0, %zmm3
	incq	%rax
	vfmadd132ps	%zmm2, %zmm0, %zmm15
	vfmadd132ps	%zmm2, %zmm0, %zmm14
	vfmadd132ps	%zmm2, %zmm0, %zmm13
	vfmadd132ps	%zmm2, %zmm0, %zmm12
	vfmadd132ps	%zmm2, %zmm0, %zmm11
	vfmadd132ps	%zmm2, %zmm0, %zmm10
	vfmadd132ps	%zmm2, %zmm0, %zmm9
	vfmadd132ps	%zmm2, %zmm0, %zmm8
	vfmadd132ps	%zmm2, %zmm0, %zmm7
	vfmadd132ps	%zmm2, %zmm0, %zmm6
	vfmadd132ps	%zmm2, %zmm0, %zmm5
	vfmadd132ps	%zmm2, %zmm0, %zmm4
	vmovaps	%zmm3, %zmm1
	cmpq	%rax, %rdx
	jne	.L339
	vmovups	%zmm15, (%rdi)
	vmovups	%zmm14, 64(%rdi)
	vmovups	%zmm13, 128(%rdi)
	vmovups	%zmm12, 192(%rdi)
	vmovups	%zmm11, 256(%rdi)
	vmovups	%zmm10, 320(%rdi)
	vmovups	%zmm9, 384(%rdi)
	vmovups	%zmm8, 448(%rdi)
	vmovups	%zmm7, 512(%rdi)
	vmovups	%zmm6, 576(%rdi)
	vmovups	%zmm5, 640(%rdi)
	vmovups	%zmm4, 704(%rdi)
	vmovups	%zmm3, 768(%rdi)
	vzeroupper
.L337:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$13>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L341:
	.cfi_restore_state
	leaq	.LC46(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L342:
	leaq	.LC46(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5752:
	.size	_Z11axpb_simd_cILi13EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi13EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi14EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC47:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 14]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi14EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi14EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi14EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi14EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi14EEl14axpb_options_tfPff:
.LFB5757:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC47(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5757:
	.size	_Z24axpb_simd_parallel_m_mnmILi14EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi14EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi14EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC48:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 14]"
	.section	.text._Z15axpb_simd_m_mnmILi14EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi14EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi14EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi14EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi14EEl14axpb_options_tfPff:
.LFB5756:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$5270498306774157605, %rdx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %r8
	movq	64(%rbp), %rsi
	movq	%r8, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%r8, %rdx
	sarq	$6, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$224, %rax, %rax
	cmpq	%rax, %r8
	jne	.L357
	cmpq	$1, 32(%rbp)
	jne	.L358
	movq	%rdi, %rcx
	testq	%r8, %r8
	leaq	15(%r8), %rdi
	cmovns	%r8, %rdi
	vmovaps	%xmm0, %xmm2
	vmovaps	%xmm1, %xmm0
	movq	%rcx, %rax
	sarq	$4, %rdi
	xorl	%ecx, %ecx
	vbroadcastss	%xmm2, %zmm1
	vbroadcastss	%xmm0, %zmm0
	cmpq	$15, %r8
	jle	.L354
.L352:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$14>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L349
	vmovups	(%rax), %zmm15
	vmovups	64(%rax), %zmm14
	vmovups	128(%rax), %zmm13
	vmovups	192(%rax), %zmm12
	vmovups	256(%rax), %zmm11
	vmovups	320(%rax), %zmm10
	vmovups	384(%rax), %zmm9
	vmovups	448(%rax), %zmm8
	vmovups	512(%rax), %zmm7
	vmovups	576(%rax), %zmm6
	vmovups	640(%rax), %zmm5
	vmovups	704(%rax), %zmm4
	vmovups	768(%rax), %zmm3
	vmovups	832(%rax), %zmm2
	xorl	%edx, %edx
	.p2align 4,,10
	.p2align 3
.L350:
	incq	%rdx
	vfmadd132ps	%zmm1, %zmm0, %zmm15
	vfmadd132ps	%zmm1, %zmm0, %zmm14
	vfmadd132ps	%zmm1, %zmm0, %zmm13
	vfmadd132ps	%zmm1, %zmm0, %zmm12
	vfmadd132ps	%zmm1, %zmm0, %zmm11
	vfmadd132ps	%zmm1, %zmm0, %zmm10
	vfmadd132ps	%zmm1, %zmm0, %zmm9
	vfmadd132ps	%zmm1, %zmm0, %zmm8
	vfmadd132ps	%zmm1, %zmm0, %zmm7
	vfmadd132ps	%zmm1, %zmm0, %zmm6
	vfmadd132ps	%zmm1, %zmm0, %zmm5
	vfmadd132ps	%zmm1, %zmm0, %zmm4
	vfmadd132ps	%zmm1, %zmm0, %zmm3
	vfmadd132ps	%zmm1, %zmm0, %zmm2
	cmpq	%rdx, %rsi
	jne	.L350
	vmovups	%zmm15, (%rax)
	vmovups	%zmm14, 64(%rax)
	vmovups	%zmm13, 128(%rax)
	vmovups	%zmm12, 192(%rax)
	vmovups	%zmm11, 256(%rax)
	vmovups	%zmm10, 320(%rax)
	vmovups	%zmm9, 384(%rax)
	vmovups	%zmm8, 448(%rax)
	vmovups	%zmm7, 512(%rax)
	vmovups	%zmm6, 576(%rax)
	vmovups	%zmm5, 640(%rax)
	vmovups	%zmm4, 704(%rax)
	vmovups	%zmm3, 768(%rax)
	vmovups	%zmm2, 832(%rax)
.L349:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$14>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$14, %rcx
	addq	$896, %rax
	cmpq	%rdi, %rcx
	jl	.L352
.L354:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L357:
	.cfi_restore_state
	leaq	.LC48(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L358:
	leaq	.LC48(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5756:
	.size	_Z15axpb_simd_m_mnmILi14EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi14EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi14EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC49:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 14]"
	.section	.text._Z11axpb_simd_cILi14EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi14EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi14EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi14EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi14EEl14axpb_options_tfPff:
.LFB5755:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$224, 56(%rbp)
	jne	.L366
	cmpq	$1, 32(%rbp)
	jne	.L367
	movq	64(%rbp), %rdx
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$14>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L362
	vbroadcastss	%xmm0, %zmm2
	vmovups	(%rdi), %zmm16
	vmovups	64(%rdi), %zmm15
	vmovups	128(%rdi), %zmm14
	vmovups	192(%rdi), %zmm13
	vmovups	256(%rdi), %zmm12
	vmovups	320(%rdi), %zmm11
	vmovups	384(%rdi), %zmm10
	vmovups	448(%rdi), %zmm9
	vmovups	512(%rdi), %zmm8
	vmovups	576(%rdi), %zmm7
	vmovups	640(%rdi), %zmm6
	vmovups	704(%rdi), %zmm5
	vmovups	768(%rdi), %zmm4
	vmovups	832(%rdi), %zmm3
	vbroadcastss	%xmm1, %zmm0
	xorl	%eax, %eax
	jmp	.L363
	.p2align 4,,10
	.p2align 3
.L364:
	vmovaps	%zmm1, %zmm3
.L363:
	vfmadd132ps	%zmm2, %zmm0, %zmm3
	incq	%rax
	vfmadd132ps	%zmm2, %zmm0, %zmm16
	vfmadd132ps	%zmm2, %zmm0, %zmm15
	vfmadd132ps	%zmm2, %zmm0, %zmm14
	vfmadd132ps	%zmm2, %zmm0, %zmm13
	vfmadd132ps	%zmm2, %zmm0, %zmm12
	vfmadd132ps	%zmm2, %zmm0, %zmm11
	vfmadd132ps	%zmm2, %zmm0, %zmm10
	vfmadd132ps	%zmm2, %zmm0, %zmm9
	vfmadd132ps	%zmm2, %zmm0, %zmm8
	vfmadd132ps	%zmm2, %zmm0, %zmm7
	vfmadd132ps	%zmm2, %zmm0, %zmm6
	vfmadd132ps	%zmm2, %zmm0, %zmm5
	vfmadd132ps	%zmm2, %zmm0, %zmm4
	vmovaps	%zmm3, %zmm1
	cmpq	%rax, %rdx
	jne	.L364
	vmovups	%zmm16, (%rdi)
	vmovups	%zmm15, 64(%rdi)
	vmovups	%zmm14, 128(%rdi)
	vmovups	%zmm13, 192(%rdi)
	vmovups	%zmm12, 256(%rdi)
	vmovups	%zmm11, 320(%rdi)
	vmovups	%zmm10, 384(%rdi)
	vmovups	%zmm9, 448(%rdi)
	vmovups	%zmm8, 512(%rdi)
	vmovups	%zmm7, 576(%rdi)
	vmovups	%zmm6, 640(%rdi)
	vmovups	%zmm5, 704(%rdi)
	vmovups	%zmm4, 768(%rdi)
	vmovups	%zmm3, 832(%rdi)
	vzeroupper
.L362:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$14>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L366:
	.cfi_restore_state
	leaq	.LC49(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L367:
	leaq	.LC49(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5755:
	.size	_Z11axpb_simd_cILi14EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi14EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi15EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC50:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 15]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi15EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi15EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi15EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi15EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi15EEl14axpb_options_tfPff:
.LFB5760:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC50(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5760:
	.size	_Z24axpb_simd_parallel_m_mnmILi15EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi15EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi15EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC51:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 15]"
	.section	.text._Z15axpb_simd_m_mnmILi15EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi15EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi15EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi15EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi15EEl14axpb_options_tfPff:
.LFB5759:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$-8608480567731124087, %rdx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %r8
	movq	64(%rbp), %rsi
	movq	%r8, %rax
	imulq	%rdx
	andq	$-64, %rsp
	leaq	(%rdx,%r8), %rax
	movq	%r8, %rdx
	sarq	$7, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$240, %rax, %rax
	cmpq	%rax, %r8
	jne	.L382
	cmpq	$1, 32(%rbp)
	jne	.L383
	movq	%rdi, %rcx
	testq	%r8, %r8
	leaq	15(%r8), %rdi
	cmovns	%r8, %rdi
	vmovaps	%xmm0, %xmm2
	vmovaps	%xmm1, %xmm0
	movq	%rcx, %rax
	sarq	$4, %rdi
	xorl	%ecx, %ecx
	vbroadcastss	%xmm2, %zmm1
	vbroadcastss	%xmm0, %zmm0
	cmpq	$15, %r8
	jle	.L379
.L377:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$15>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L374
	vmovups	(%rax), %zmm16
	vmovups	64(%rax), %zmm15
	vmovups	128(%rax), %zmm14
	vmovups	192(%rax), %zmm13
	vmovups	256(%rax), %zmm12
	vmovups	320(%rax), %zmm11
	vmovups	384(%rax), %zmm10
	vmovups	448(%rax), %zmm9
	vmovups	512(%rax), %zmm8
	vmovups	576(%rax), %zmm7
	vmovups	640(%rax), %zmm6
	vmovups	704(%rax), %zmm5
	vmovups	768(%rax), %zmm4
	vmovups	832(%rax), %zmm3
	vmovups	896(%rax), %zmm2
	xorl	%edx, %edx
	.p2align 4,,10
	.p2align 3
.L375:
	incq	%rdx
	vfmadd132ps	%zmm1, %zmm0, %zmm16
	vfmadd132ps	%zmm1, %zmm0, %zmm15
	vfmadd132ps	%zmm1, %zmm0, %zmm14
	vfmadd132ps	%zmm1, %zmm0, %zmm13
	vfmadd132ps	%zmm1, %zmm0, %zmm12
	vfmadd132ps	%zmm1, %zmm0, %zmm11
	vfmadd132ps	%zmm1, %zmm0, %zmm10
	vfmadd132ps	%zmm1, %zmm0, %zmm9
	vfmadd132ps	%zmm1, %zmm0, %zmm8
	vfmadd132ps	%zmm1, %zmm0, %zmm7
	vfmadd132ps	%zmm1, %zmm0, %zmm6
	vfmadd132ps	%zmm1, %zmm0, %zmm5
	vfmadd132ps	%zmm1, %zmm0, %zmm4
	vfmadd132ps	%zmm1, %zmm0, %zmm3
	vfmadd132ps	%zmm1, %zmm0, %zmm2
	cmpq	%rdx, %rsi
	jne	.L375
	vmovups	%zmm16, (%rax)
	vmovups	%zmm15, 64(%rax)
	vmovups	%zmm14, 128(%rax)
	vmovups	%zmm13, 192(%rax)
	vmovups	%zmm12, 256(%rax)
	vmovups	%zmm11, 320(%rax)
	vmovups	%zmm10, 384(%rax)
	vmovups	%zmm9, 448(%rax)
	vmovups	%zmm8, 512(%rax)
	vmovups	%zmm7, 576(%rax)
	vmovups	%zmm6, 640(%rax)
	vmovups	%zmm5, 704(%rax)
	vmovups	%zmm4, 768(%rax)
	vmovups	%zmm3, 832(%rax)
	vmovups	%zmm2, 896(%rax)
.L374:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$15>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$15, %rcx
	addq	$960, %rax
	cmpq	%rdi, %rcx
	jl	.L377
.L379:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L382:
	.cfi_restore_state
	leaq	.LC51(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L383:
	leaq	.LC51(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5759:
	.size	_Z15axpb_simd_m_mnmILi15EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi15EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi15EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC52:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 15]"
	.section	.text._Z11axpb_simd_cILi15EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi15EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi15EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi15EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi15EEl14axpb_options_tfPff:
.LFB5758:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$240, 56(%rbp)
	jne	.L391
	cmpq	$1, 32(%rbp)
	jne	.L392
	movq	64(%rbp), %rdx
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$15>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L387
	vbroadcastss	%xmm0, %zmm2
	vmovups	(%rdi), %zmm17
	vmovups	64(%rdi), %zmm16
	vmovups	128(%rdi), %zmm15
	vmovups	192(%rdi), %zmm14
	vmovups	256(%rdi), %zmm13
	vmovups	320(%rdi), %zmm12
	vmovups	384(%rdi), %zmm11
	vmovups	448(%rdi), %zmm10
	vmovups	512(%rdi), %zmm9
	vmovups	576(%rdi), %zmm8
	vmovups	640(%rdi), %zmm7
	vmovups	704(%rdi), %zmm6
	vmovups	768(%rdi), %zmm5
	vmovups	832(%rdi), %zmm4
	vmovups	896(%rdi), %zmm3
	vbroadcastss	%xmm1, %zmm0
	xorl	%eax, %eax
	jmp	.L388
	.p2align 4,,10
	.p2align 3
.L389:
	vmovaps	%zmm1, %zmm3
.L388:
	vfmadd132ps	%zmm2, %zmm0, %zmm3
	incq	%rax
	vfmadd132ps	%zmm2, %zmm0, %zmm17
	vfmadd132ps	%zmm2, %zmm0, %zmm16
	vfmadd132ps	%zmm2, %zmm0, %zmm15
	vfmadd132ps	%zmm2, %zmm0, %zmm14
	vfmadd132ps	%zmm2, %zmm0, %zmm13
	vfmadd132ps	%zmm2, %zmm0, %zmm12
	vfmadd132ps	%zmm2, %zmm0, %zmm11
	vfmadd132ps	%zmm2, %zmm0, %zmm10
	vfmadd132ps	%zmm2, %zmm0, %zmm9
	vfmadd132ps	%zmm2, %zmm0, %zmm8
	vfmadd132ps	%zmm2, %zmm0, %zmm7
	vfmadd132ps	%zmm2, %zmm0, %zmm6
	vfmadd132ps	%zmm2, %zmm0, %zmm5
	vfmadd132ps	%zmm2, %zmm0, %zmm4
	vmovaps	%zmm3, %zmm1
	cmpq	%rax, %rdx
	jne	.L389
	vmovups	%zmm17, (%rdi)
	vmovups	%zmm16, 64(%rdi)
	vmovups	%zmm15, 128(%rdi)
	vmovups	%zmm14, 192(%rdi)
	vmovups	%zmm13, 256(%rdi)
	vmovups	%zmm12, 320(%rdi)
	vmovups	%zmm11, 384(%rdi)
	vmovups	%zmm10, 448(%rdi)
	vmovups	%zmm9, 512(%rdi)
	vmovups	%zmm8, 576(%rdi)
	vmovups	%zmm7, 640(%rdi)
	vmovups	%zmm6, 704(%rdi)
	vmovups	%zmm5, 768(%rdi)
	vmovups	%zmm4, 832(%rdi)
	vmovups	%zmm3, 896(%rdi)
	vzeroupper
.L387:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$15>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L391:
	.cfi_restore_state
	leaq	.LC52(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L392:
	leaq	.LC52(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5758:
	.size	_Z11axpb_simd_cILi15EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi15EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi16EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC53:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 16]"
	.section	.text._Z15axpb_simd_m_mnmILi16EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi16EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi16EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi16EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi16EEl14axpb_options_tfPff:
.LFB5762:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rdx
	movq	64(%rbp), %rcx
	andq	$-64, %rsp
	testb	%dl, %dl
	jne	.L405
	cmpq	$1, 32(%rbp)
	jne	.L406
	testq	%rdx, %rdx
	movq	%rdi, %rax
	leaq	15(%rdx), %rdi
	cmovns	%rdx, %rdi
	vmovaps	%xmm0, %xmm2
	vmovaps	%xmm1, %xmm0
	sarq	$4, %rdi
	xorl	%esi, %esi
	vbroadcastss	%xmm2, %zmm1
	vbroadcastss	%xmm0, %zmm0
	cmpq	$15, %rdx
	jle	.L402
.L400:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$16>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rcx, %rcx
	jle	.L397
	vmovups	(%rax), %zmm17
	vmovups	64(%rax), %zmm16
	vmovups	128(%rax), %zmm15
	vmovups	192(%rax), %zmm14
	vmovups	256(%rax), %zmm13
	vmovups	320(%rax), %zmm12
	vmovups	384(%rax), %zmm11
	vmovups	448(%rax), %zmm10
	vmovups	512(%rax), %zmm9
	vmovups	576(%rax), %zmm8
	vmovups	640(%rax), %zmm7
	vmovups	704(%rax), %zmm6
	vmovups	768(%rax), %zmm5
	vmovups	832(%rax), %zmm4
	vmovups	896(%rax), %zmm3
	vmovups	960(%rax), %zmm2
	xorl	%edx, %edx
	.p2align 4,,10
	.p2align 3
.L398:
	incq	%rdx
	vfmadd132ps	%zmm1, %zmm0, %zmm17
	vfmadd132ps	%zmm1, %zmm0, %zmm16
	vfmadd132ps	%zmm1, %zmm0, %zmm15
	vfmadd132ps	%zmm1, %zmm0, %zmm14
	vfmadd132ps	%zmm1, %zmm0, %zmm13
	vfmadd132ps	%zmm1, %zmm0, %zmm12
	vfmadd132ps	%zmm1, %zmm0, %zmm11
	vfmadd132ps	%zmm1, %zmm0, %zmm10
	vfmadd132ps	%zmm1, %zmm0, %zmm9
	vfmadd132ps	%zmm1, %zmm0, %zmm8
	vfmadd132ps	%zmm1, %zmm0, %zmm7
	vfmadd132ps	%zmm1, %zmm0, %zmm6
	vfmadd132ps	%zmm1, %zmm0, %zmm5
	vfmadd132ps	%zmm1, %zmm0, %zmm4
	vfmadd132ps	%zmm1, %zmm0, %zmm3
	vfmadd132ps	%zmm1, %zmm0, %zmm2
	cmpq	%rdx, %rcx
	jne	.L398
	vmovups	%zmm17, (%rax)
	vmovups	%zmm16, 64(%rax)
	vmovups	%zmm15, 128(%rax)
	vmovups	%zmm14, 192(%rax)
	vmovups	%zmm13, 256(%rax)
	vmovups	%zmm12, 320(%rax)
	vmovups	%zmm11, 384(%rax)
	vmovups	%zmm10, 448(%rax)
	vmovups	%zmm9, 512(%rax)
	vmovups	%zmm8, 576(%rax)
	vmovups	%zmm7, 640(%rax)
	vmovups	%zmm6, 704(%rax)
	vmovups	%zmm5, 768(%rax)
	vmovups	%zmm4, 832(%rax)
	vmovups	%zmm3, 896(%rax)
	vmovups	%zmm2, 960(%rax)
.L397:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$16>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$16, %rsi
	addq	$1024, %rax
	cmpq	%rdi, %rsi
	jl	.L400
.L402:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L405:
	.cfi_restore_state
	leaq	.LC53(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L406:
	leaq	.LC53(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5762:
	.size	_Z15axpb_simd_m_mnmILi16EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi16EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi16EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC54:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 16]"
	.section	.text._Z11axpb_simd_cILi16EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi16EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi16EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi16EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi16EEl14axpb_options_tfPff:
.LFB5761:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$256, 56(%rbp)
	jne	.L414
	cmpq	$1, 32(%rbp)
	jne	.L415
	movq	64(%rbp), %rdx
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$16>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L410
	vmovaps	%xmm0, %xmm2
	vmovups	(%rdi), %zmm17
	vmovaps	%xmm1, %xmm0
	vmovups	64(%rdi), %zmm16
	vmovups	128(%rdi), %zmm15
	vmovups	192(%rdi), %zmm14
	vmovups	256(%rdi), %zmm13
	vmovups	320(%rdi), %zmm12
	vmovups	384(%rdi), %zmm11
	vmovups	448(%rdi), %zmm10
	vmovups	512(%rdi), %zmm9
	vmovups	576(%rdi), %zmm8
	vmovups	640(%rdi), %zmm7
	vmovups	704(%rdi), %zmm6
	vmovups	768(%rdi), %zmm5
	vmovups	832(%rdi), %zmm4
	vmovups	896(%rdi), %zmm3
	vmovups	960(%rdi), %zmm1
	vbroadcastss	%xmm2, %zmm2
	vbroadcastss	%xmm0, %zmm0
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L411:
	incq	%rax
	vfmadd132ps	%zmm2, %zmm0, %zmm17
	vfmadd132ps	%zmm2, %zmm0, %zmm16
	vfmadd132ps	%zmm2, %zmm0, %zmm15
	vfmadd132ps	%zmm2, %zmm0, %zmm14
	vfmadd132ps	%zmm2, %zmm0, %zmm13
	vfmadd132ps	%zmm2, %zmm0, %zmm12
	vfmadd132ps	%zmm2, %zmm0, %zmm11
	vfmadd132ps	%zmm2, %zmm0, %zmm10
	vfmadd132ps	%zmm2, %zmm0, %zmm9
	vfmadd132ps	%zmm2, %zmm0, %zmm8
	vfmadd132ps	%zmm2, %zmm0, %zmm7
	vfmadd132ps	%zmm2, %zmm0, %zmm6
	vfmadd132ps	%zmm2, %zmm0, %zmm5
	vfmadd132ps	%zmm2, %zmm0, %zmm4
	vfmadd132ps	%zmm2, %zmm0, %zmm3
	vfmadd132ps	%zmm2, %zmm0, %zmm1
	cmpq	%rax, %rdx
	jne	.L411
	vmovups	%zmm17, (%rdi)
	vmovups	%zmm16, 64(%rdi)
	vmovups	%zmm15, 128(%rdi)
	vmovups	%zmm14, 192(%rdi)
	vmovups	%zmm13, 256(%rdi)
	vmovups	%zmm12, 320(%rdi)
	vmovups	%zmm11, 384(%rdi)
	vmovups	%zmm10, 448(%rdi)
	vmovups	%zmm9, 512(%rdi)
	vmovups	%zmm8, 576(%rdi)
	vmovups	%zmm7, 640(%rdi)
	vmovups	%zmm6, 704(%rdi)
	vmovups	%zmm5, 768(%rdi)
	vmovups	%zmm4, 832(%rdi)
	vmovups	%zmm3, 896(%rdi)
	vmovups	%zmm1, 960(%rdi)
	vzeroupper
.L410:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$16>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L414:
	.cfi_restore_state
	leaq	.LC54(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L415:
	leaq	.LC54(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5761:
	.size	_Z11axpb_simd_cILi16EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi16EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi17EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC55:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 17]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi17EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi17EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi17EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi17EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi17EEl14axpb_options_tfPff:
.LFB5766:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC55(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5766:
	.size	_Z24axpb_simd_parallel_m_mnmILi17EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi17EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi17EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC56:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 17]"
	.section	.text._Z15axpb_simd_m_mnmILi17EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi17EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi17EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi17EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi17EEl14axpb_options_tfPff:
.LFB5765:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$8680820740569200761, %rdx
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %r8
	movq	64(%rbp), %rsi
	movq	%r8, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%r8, %rdx
	sarq	$7, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$272, %rax, %rax
	cmpq	%rax, %r8
	jne	.L434
	cmpq	$1, 32(%rbp)
	jne	.L435
	movq	%rdi, %rcx
	testq	%r8, %r8
	leaq	15(%r8), %rdi
	cmovns	%r8, %rdi
	movq	%rcx, %rax
	sarq	$4, %rdi
	xorl	%ecx, %ecx
	cmpq	$15, %r8
	jle	.L430
	vmovaps	%xmm0, %xmm2
	leaq	-3(%rsi), %r9
	vmovaps	%xmm1, %xmm0
	vbroadcastss	%xmm0, %zmm0
	vbroadcastss	%xmm2, %zmm1
	andq	$-2, %r9
.L424:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$17>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L427
	xorl	%edx, %edx
	cmpq	$2, %rsi
	jle	.L425
	vmovups	(%rax), %zmm18
	vmovups	64(%rax), %zmm17
	vmovups	128(%rax), %zmm16
	vmovups	192(%rax), %zmm15
	vmovups	256(%rax), %zmm14
	vmovups	320(%rax), %zmm13
	vmovups	384(%rax), %zmm12
	vmovups	448(%rax), %zmm11
	vmovups	512(%rax), %zmm10
	vmovups	576(%rax), %zmm9
	vmovups	640(%rax), %zmm8
	vmovups	704(%rax), %zmm7
	vmovups	768(%rax), %zmm6
	vmovups	832(%rax), %zmm5
	vmovups	896(%rax), %zmm4
	vmovups	960(%rax), %zmm3
	vmovups	1024(%rax), %zmm2
.L426:
	vfmadd132ps	%zmm1, %zmm0, %zmm18
	vfmadd132ps	%zmm1, %zmm0, %zmm17
	vfmadd132ps	%zmm1, %zmm0, %zmm16
	vfmadd132ps	%zmm1, %zmm0, %zmm15
	vfmadd132ps	%zmm1, %zmm0, %zmm14
	vfmadd132ps	%zmm1, %zmm0, %zmm13
	vfmadd132ps	%zmm1, %zmm0, %zmm12
	vfmadd132ps	%zmm1, %zmm0, %zmm11
	vfmadd132ps	%zmm1, %zmm0, %zmm10
	vfmadd132ps	%zmm1, %zmm0, %zmm9
	vfmadd132ps	%zmm1, %zmm0, %zmm8
	vfmadd132ps	%zmm1, %zmm0, %zmm7
	vfmadd132ps	%zmm1, %zmm0, %zmm6
	vfmadd132ps	%zmm1, %zmm0, %zmm5
	vfmadd132ps	%zmm1, %zmm0, %zmm4
	vfmadd132ps	%zmm1, %zmm0, %zmm3
	vfmadd132ps	%zmm1, %zmm0, %zmm2
	vfmadd132ps	%zmm1, %zmm0, %zmm18
	vfmadd132ps	%zmm1, %zmm0, %zmm17
	vfmadd132ps	%zmm1, %zmm0, %zmm16
	vfmadd132ps	%zmm1, %zmm0, %zmm15
	vfmadd132ps	%zmm1, %zmm0, %zmm14
	vfmadd132ps	%zmm1, %zmm0, %zmm13
	vfmadd132ps	%zmm1, %zmm0, %zmm12
	vfmadd132ps	%zmm1, %zmm0, %zmm11
	vfmadd132ps	%zmm1, %zmm0, %zmm10
	vfmadd132ps	%zmm1, %zmm0, %zmm9
	vfmadd132ps	%zmm1, %zmm0, %zmm8
	vfmadd132ps	%zmm1, %zmm0, %zmm7
	vfmadd132ps	%zmm1, %zmm0, %zmm6
	vfmadd132ps	%zmm1, %zmm0, %zmm5
	vfmadd132ps	%zmm1, %zmm0, %zmm4
	vfmadd132ps	%zmm1, %zmm0, %zmm3
	vfmadd132ps	%zmm1, %zmm0, %zmm2
	movq	%rdx, %r8
	vmovups	%zmm18, (%rax)
	vmovups	%zmm17, 64(%rax)
	vmovups	%zmm16, 128(%rax)
	vmovups	%zmm15, 192(%rax)
	vmovups	%zmm14, 256(%rax)
	vmovups	%zmm13, 320(%rax)
	vmovups	%zmm12, 384(%rax)
	vmovups	%zmm11, 448(%rax)
	vmovups	%zmm10, 512(%rax)
	vmovups	%zmm9, 576(%rax)
	vmovups	%zmm8, 640(%rax)
	vmovups	%zmm7, 704(%rax)
	vmovups	%zmm6, 768(%rax)
	vmovups	%zmm5, 832(%rax)
	vmovups	%zmm4, 896(%rax)
	vmovups	%zmm3, 960(%rax)
	vmovups	%zmm2, 1024(%rax)
	addq	$2, %rdx
	cmpq	%r9, %r8
	jne	.L426
.L425:
	vmovups	(%rax), %zmm18
	vmovups	64(%rax), %zmm17
	vmovups	128(%rax), %zmm16
	vmovups	192(%rax), %zmm15
	vmovups	256(%rax), %zmm14
	vmovups	320(%rax), %zmm13
	vmovups	384(%rax), %zmm12
	vmovups	448(%rax), %zmm11
	vmovups	512(%rax), %zmm10
	vmovups	576(%rax), %zmm9
	vmovups	640(%rax), %zmm8
	vmovups	704(%rax), %zmm7
	vmovups	768(%rax), %zmm6
	vmovups	832(%rax), %zmm5
	vmovups	896(%rax), %zmm4
	vmovups	960(%rax), %zmm3
	vmovups	1024(%rax), %zmm2
	.p2align 4,,10
	.p2align 3
.L428:
	vfmadd132ps	%zmm1, %zmm0, %zmm18
	vfmadd132ps	%zmm1, %zmm0, %zmm17
	vfmadd132ps	%zmm1, %zmm0, %zmm16
	vfmadd132ps	%zmm1, %zmm0, %zmm15
	vfmadd132ps	%zmm1, %zmm0, %zmm14
	vfmadd132ps	%zmm1, %zmm0, %zmm13
	vfmadd132ps	%zmm1, %zmm0, %zmm12
	vfmadd132ps	%zmm1, %zmm0, %zmm11
	vfmadd132ps	%zmm1, %zmm0, %zmm10
	vfmadd132ps	%zmm1, %zmm0, %zmm9
	vfmadd132ps	%zmm1, %zmm0, %zmm8
	vfmadd132ps	%zmm1, %zmm0, %zmm7
	vfmadd132ps	%zmm1, %zmm0, %zmm6
	vfmadd132ps	%zmm1, %zmm0, %zmm5
	vfmadd132ps	%zmm1, %zmm0, %zmm4
	vfmadd132ps	%zmm1, %zmm0, %zmm3
	vfmadd132ps	%zmm1, %zmm0, %zmm2
	incq	%rdx
	vmovups	%zmm18, (%rax)
	vmovups	%zmm17, 64(%rax)
	vmovups	%zmm16, 128(%rax)
	vmovups	%zmm15, 192(%rax)
	vmovups	%zmm14, 256(%rax)
	vmovups	%zmm13, 320(%rax)
	vmovups	%zmm12, 384(%rax)
	vmovups	%zmm11, 448(%rax)
	vmovups	%zmm10, 512(%rax)
	vmovups	%zmm9, 576(%rax)
	vmovups	%zmm8, 640(%rax)
	vmovups	%zmm7, 704(%rax)
	vmovups	%zmm6, 768(%rax)
	vmovups	%zmm5, 832(%rax)
	vmovups	%zmm4, 896(%rax)
	vmovups	%zmm3, 960(%rax)
	vmovups	%zmm2, 1024(%rax)
	cmpq	%rdx, %rsi
	jg	.L428
.L427:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$17>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$17, %rcx
	addq	$1088, %rax
	cmpq	%rcx, %rdi
	jg	.L424
	vzeroupper
.L430:
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L434:
	.cfi_restore_state
	leaq	.LC56(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L435:
	leaq	.LC56(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5765:
	.size	_Z15axpb_simd_m_mnmILi17EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi17EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi17EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC57:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 17]"
	.section	.text._Z11axpb_simd_cILi17EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi17EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi17EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi17EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi17EEl14axpb_options_tfPff:
.LFB5764:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$272, 56(%rbp)
	jne	.L449
	cmpq	$1, 32(%rbp)
	jne	.L450
	movq	64(%rbp), %rdx
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$17>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L443
	vbroadcastss	%xmm0, %zmm0
	vbroadcastss	%xmm1, %zmm1
	cmpq	$2, %rdx
	jle	.L444
	leaq	-3(%rdx), %rcx
	vmovups	(%rdi), %zmm18
	vmovups	64(%rdi), %zmm17
	vmovups	128(%rdi), %zmm16
	vmovups	192(%rdi), %zmm15
	vmovups	256(%rdi), %zmm14
	vmovups	320(%rdi), %zmm13
	vmovups	384(%rdi), %zmm12
	vmovups	448(%rdi), %zmm11
	vmovups	512(%rdi), %zmm10
	vmovups	576(%rdi), %zmm9
	vmovups	640(%rdi), %zmm8
	vmovups	704(%rdi), %zmm7
	vmovups	768(%rdi), %zmm6
	vmovups	832(%rdi), %zmm5
	vmovups	896(%rdi), %zmm4
	vmovups	960(%rdi), %zmm3
	vmovups	1024(%rdi), %zmm2
	andq	$-2, %rcx
	xorl	%eax, %eax
.L441:
	vfmadd132ps	%zmm0, %zmm1, %zmm18
	vfmadd132ps	%zmm0, %zmm1, %zmm17
	vfmadd132ps	%zmm0, %zmm1, %zmm16
	vfmadd132ps	%zmm0, %zmm1, %zmm15
	vfmadd132ps	%zmm0, %zmm1, %zmm14
	vfmadd132ps	%zmm0, %zmm1, %zmm13
	vfmadd132ps	%zmm0, %zmm1, %zmm12
	vfmadd132ps	%zmm0, %zmm1, %zmm11
	vfmadd132ps	%zmm0, %zmm1, %zmm10
	vfmadd132ps	%zmm0, %zmm1, %zmm9
	vfmadd132ps	%zmm0, %zmm1, %zmm8
	vfmadd132ps	%zmm0, %zmm1, %zmm7
	vfmadd132ps	%zmm0, %zmm1, %zmm6
	vfmadd132ps	%zmm0, %zmm1, %zmm5
	vfmadd132ps	%zmm0, %zmm1, %zmm4
	vfmadd132ps	%zmm0, %zmm1, %zmm3
	vfmadd132ps	%zmm0, %zmm1, %zmm2
	movq	%rax, %rsi
	vfmadd132ps	%zmm0, %zmm1, %zmm18
	vfmadd132ps	%zmm0, %zmm1, %zmm17
	vfmadd132ps	%zmm0, %zmm1, %zmm16
	vfmadd132ps	%zmm0, %zmm1, %zmm15
	vfmadd132ps	%zmm0, %zmm1, %zmm14
	vfmadd132ps	%zmm0, %zmm1, %zmm13
	vfmadd132ps	%zmm0, %zmm1, %zmm12
	vfmadd132ps	%zmm0, %zmm1, %zmm11
	vfmadd132ps	%zmm0, %zmm1, %zmm10
	vfmadd132ps	%zmm0, %zmm1, %zmm9
	vfmadd132ps	%zmm0, %zmm1, %zmm8
	vfmadd132ps	%zmm0, %zmm1, %zmm7
	vfmadd132ps	%zmm0, %zmm1, %zmm6
	vfmadd132ps	%zmm0, %zmm1, %zmm5
	vfmadd132ps	%zmm0, %zmm1, %zmm4
	vfmadd132ps	%zmm0, %zmm1, %zmm3
	vfmadd132ps	%zmm0, %zmm1, %zmm2
	addq	$2, %rax
	cmpq	%rcx, %rsi
	jne	.L441
	vmovups	%zmm18, (%rdi)
	vmovups	%zmm17, 64(%rdi)
	vmovups	%zmm16, 128(%rdi)
	vmovups	%zmm15, 192(%rdi)
	vmovups	%zmm14, 256(%rdi)
	vmovups	%zmm13, 320(%rdi)
	vmovups	%zmm12, 384(%rdi)
	vmovups	%zmm11, 448(%rdi)
	vmovups	%zmm10, 512(%rdi)
	vmovups	%zmm9, 576(%rdi)
	vmovups	%zmm8, 640(%rdi)
	vmovups	%zmm7, 704(%rdi)
	vmovups	%zmm6, 768(%rdi)
	vmovups	%zmm5, 832(%rdi)
	vmovups	%zmm4, 896(%rdi)
	vmovups	%zmm3, 960(%rdi)
	vmovups	%zmm2, 1024(%rdi)
.L440:
	vmovups	(%rdi), %zmm18
	vmovups	64(%rdi), %zmm17
	vmovups	128(%rdi), %zmm16
	vmovups	192(%rdi), %zmm15
	vmovups	256(%rdi), %zmm14
	vmovups	320(%rdi), %zmm13
	vmovups	384(%rdi), %zmm12
	vmovups	448(%rdi), %zmm11
	vmovups	512(%rdi), %zmm10
	vmovups	576(%rdi), %zmm9
	vmovups	640(%rdi), %zmm8
	vmovups	704(%rdi), %zmm7
	vmovups	768(%rdi), %zmm6
	vmovups	832(%rdi), %zmm5
	vmovups	896(%rdi), %zmm4
	vmovups	960(%rdi), %zmm3
	vmovups	1024(%rdi), %zmm2
	.p2align 4,,10
	.p2align 3
.L442:
	incq	%rax
	vfmadd132ps	%zmm0, %zmm1, %zmm18
	vfmadd132ps	%zmm0, %zmm1, %zmm17
	vfmadd132ps	%zmm0, %zmm1, %zmm16
	vfmadd132ps	%zmm0, %zmm1, %zmm15
	vfmadd132ps	%zmm0, %zmm1, %zmm14
	vfmadd132ps	%zmm0, %zmm1, %zmm13
	vfmadd132ps	%zmm0, %zmm1, %zmm12
	vfmadd132ps	%zmm0, %zmm1, %zmm11
	vfmadd132ps	%zmm0, %zmm1, %zmm10
	vfmadd132ps	%zmm0, %zmm1, %zmm9
	vfmadd132ps	%zmm0, %zmm1, %zmm8
	vfmadd132ps	%zmm0, %zmm1, %zmm7
	vfmadd132ps	%zmm0, %zmm1, %zmm6
	vfmadd132ps	%zmm0, %zmm1, %zmm5
	vfmadd132ps	%zmm0, %zmm1, %zmm4
	vfmadd132ps	%zmm0, %zmm1, %zmm3
	vfmadd132ps	%zmm0, %zmm1, %zmm2
	cmpq	%rax, %rdx
	jg	.L442
	vmovups	%zmm18, (%rdi)
	vmovups	%zmm17, 64(%rdi)
	vmovups	%zmm16, 128(%rdi)
	vmovups	%zmm15, 192(%rdi)
	vmovups	%zmm14, 256(%rdi)
	vmovups	%zmm13, 320(%rdi)
	vmovups	%zmm12, 384(%rdi)
	vmovups	%zmm11, 448(%rdi)
	vmovups	%zmm10, 512(%rdi)
	vmovups	%zmm9, 576(%rdi)
	vmovups	%zmm8, 640(%rdi)
	vmovups	%zmm7, 704(%rdi)
	vmovups	%zmm6, 768(%rdi)
	vmovups	%zmm5, 832(%rdi)
	vmovups	%zmm4, 896(%rdi)
	vmovups	%zmm3, 960(%rdi)
	vmovups	%zmm2, 1024(%rdi)
	vzeroupper
.L443:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$17>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L444:
	.cfi_restore_state
	xorl	%eax, %eax
	jmp	.L440
.L449:
	leaq	.LC57(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L450:
	leaq	.LC57(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5764:
	.size	_Z11axpb_simd_cILi17EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi17EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi18EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC58:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 18]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi18EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi18EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi18EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi18EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi18EEl14axpb_options_tfPff:
.LFB5769:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC58(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5769:
	.size	_Z24axpb_simd_parallel_m_mnmILi18EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi18EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi18EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC59:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 18]"
	.section	.text._Z15axpb_simd_m_mnmILi18EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi18EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi18EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi18EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi18EEl14axpb_options_tfPff:
.LFB5768:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$1024819115206086201, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$4, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	leaq	(%rax,%rax,8), %rax
	salq	$5, %rax
	cmpq	%rax, %rcx
	jne	.L473
	cmpq	$1, 32(%rbp)
	jne	.L474
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	1152(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L467
.L459:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$18>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L464
	cmpq	$2, %rdi
	jle	.L466
	movl	$1, %r8d
.L462:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L461:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L461
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L462
	.p2align 4,,10
	.p2align 3
.L465:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L463:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L463
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L465
.L464:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$18>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$18, %r9
	addq	$1152, %rsi
	addq	$1152, %rdx
	cmpq	%r9, %r10
	jg	.L459
.L467:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L466:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L465
.L473:
	leaq	.LC59(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L474:
	leaq	.LC59(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5768:
	.size	_Z15axpb_simd_m_mnmILi18EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi18EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi18EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC60:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 18]"
	.section	.text._Z11axpb_simd_cILi18EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi18EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi18EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi18EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi18EEl14axpb_options_tfPff:
.LFB5767:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$288, 56(%rbp)
	jne	.L491
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L492
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$18>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L483
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L485
	leaq	1152(%rdi), %rdx
.L481:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L480:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L480
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L481
	.p2align 4,,10
	.p2align 3
.L484:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L482:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L482
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L484
	vzeroupper
.L483:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$18>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L485:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	1152(%rdi), %rdx
	jmp	.L484
.L491:
	leaq	.LC60(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L492:
	leaq	.LC60(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5767:
	.size	_Z11axpb_simd_cILi18EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi18EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi19EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC61:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 19]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi19EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi19EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi19EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi19EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi19EEl14axpb_options_tfPff:
.LFB5772:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC61(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5772:
	.size	_Z24axpb_simd_parallel_m_mnmILi19EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi19EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi19EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC62:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 19]"
	.section	.text._Z15axpb_simd_m_mnmILi19EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi19EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi19EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi19EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi19EEl14axpb_options_tfPff:
.LFB5771:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$485440633518672411, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$3, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$304, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L515
	cmpq	$1, 32(%rbp)
	jne	.L516
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	1216(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L509
.L501:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$19>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L506
	cmpq	$2, %rdi
	jle	.L508
	movl	$1, %r8d
.L504:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L503:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L503
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L504
	.p2align 4,,10
	.p2align 3
.L507:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L505:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L505
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L507
.L506:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$19>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$19, %r9
	addq	$1216, %rsi
	addq	$1216, %rdx
	cmpq	%r9, %r10
	jg	.L501
.L509:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L508:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L507
.L515:
	leaq	.LC62(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L516:
	leaq	.LC62(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5771:
	.size	_Z15axpb_simd_m_mnmILi19EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi19EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi19EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC63:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 19]"
	.section	.text._Z11axpb_simd_cILi19EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi19EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi19EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi19EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi19EEl14axpb_options_tfPff:
.LFB5770:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$304, 56(%rbp)
	jne	.L533
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L534
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$19>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L525
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L527
	leaq	1216(%rdi), %rdx
.L523:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L522:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L522
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L523
	.p2align 4,,10
	.p2align 3
.L526:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L524:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L524
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L526
	vzeroupper
.L525:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$19>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L527:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	1216(%rdi), %rdx
	jmp	.L526
.L533:
	leaq	.LC63(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L534:
	leaq	.LC63(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5770:
	.size	_Z11axpb_simd_cILi19EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi19EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi20EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC64:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 20]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi20EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi20EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi20EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi20EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi20EEl14axpb_options_tfPff:
.LFB5775:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC64(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5775:
	.size	_Z24axpb_simd_parallel_m_mnmILi20EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi20EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi20EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC65:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 20]"
	.section	.text._Z15axpb_simd_m_mnmILi20EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi20EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi20EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi20EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi20EEl14axpb_options_tfPff:
.LFB5774:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$7378697629483820647, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$7, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	leaq	(%rax,%rax,4), %rax
	salq	$6, %rax
	cmpq	%rax, %rcx
	jne	.L557
	cmpq	$1, 32(%rbp)
	jne	.L558
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	1280(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L551
.L543:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$20>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L548
	cmpq	$2, %rdi
	jle	.L550
	movl	$1, %r8d
.L546:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L545:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L545
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L546
	.p2align 4,,10
	.p2align 3
.L549:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L547:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L547
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L549
.L548:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$20>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$20, %r9
	addq	$1280, %rsi
	addq	$1280, %rdx
	cmpq	%r9, %r10
	jg	.L543
.L551:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L550:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L549
.L557:
	leaq	.LC65(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L558:
	leaq	.LC65(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5774:
	.size	_Z15axpb_simd_m_mnmILi20EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi20EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi20EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC66:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 20]"
	.section	.text._Z11axpb_simd_cILi20EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi20EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi20EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi20EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi20EEl14axpb_options_tfPff:
.LFB5773:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$320, 56(%rbp)
	jne	.L575
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L576
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$20>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L567
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L569
	leaq	1280(%rdi), %rdx
.L565:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L564:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L564
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L565
	.p2align 4,,10
	.p2align 3
.L568:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L566:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L566
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L568
	vzeroupper
.L567:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$20>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L569:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	1280(%rdi), %rdx
	jmp	.L568
.L575:
	leaq	.LC66(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L576:
	leaq	.LC66(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5773:
	.size	_Z11axpb_simd_cILi20EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi20EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi21EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC67:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 21]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi21EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi21EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi21EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi21EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi21EEl14axpb_options_tfPff:
.LFB5778:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC67(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5778:
	.size	_Z24axpb_simd_parallel_m_mnmILi21EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi21EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi21EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC68:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 21]"
	.section	.text._Z15axpb_simd_m_mnmILi21EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi21EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi21EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi21EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi21EEl14axpb_options_tfPff:
.LFB5777:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$-4392081922311798003, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	leaq	(%rdx,%rcx), %rax
	movq	%rcx, %rdx
	sarq	$8, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$336, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L599
	cmpq	$1, 32(%rbp)
	jne	.L600
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	1344(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L593
.L585:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$21>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L590
	cmpq	$2, %rdi
	jle	.L592
	movl	$1, %r8d
.L588:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L587:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L587
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L588
	.p2align 4,,10
	.p2align 3
.L591:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L589:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L589
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L591
.L590:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$21>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$21, %r9
	addq	$1344, %rsi
	addq	$1344, %rdx
	cmpq	%r9, %r10
	jg	.L585
.L593:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L592:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L591
.L599:
	leaq	.LC68(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L600:
	leaq	.LC68(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5777:
	.size	_Z15axpb_simd_m_mnmILi21EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi21EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi21EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC69:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 21]"
	.section	.text._Z11axpb_simd_cILi21EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi21EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi21EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi21EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi21EEl14axpb_options_tfPff:
.LFB5776:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$336, 56(%rbp)
	jne	.L617
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L618
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$21>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L609
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L611
	leaq	1344(%rdi), %rdx
.L607:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L606:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L606
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L607
	.p2align 4,,10
	.p2align 3
.L610:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L608:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L608
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L610
	vzeroupper
.L609:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$21>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L611:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	1344(%rdi), %rdx
	jmp	.L610
.L617:
	leaq	.LC69(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L618:
	leaq	.LC69(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5776:
	.size	_Z11axpb_simd_cILi21EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi21EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi22EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC70:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 22]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi22EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi22EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi22EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi22EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi22EEl14axpb_options_tfPff:
.LFB5781:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC70(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5781:
	.size	_Z24axpb_simd_parallel_m_mnmILi22EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi22EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi22EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC71:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 22]"
	.section	.text._Z15axpb_simd_m_mnmILi22EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi22EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi22EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi22EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi22EEl14axpb_options_tfPff:
.LFB5780:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$3353953467947191203, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$6, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$352, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L641
	cmpq	$1, 32(%rbp)
	jne	.L642
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	1408(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L635
.L627:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$22>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L632
	cmpq	$2, %rdi
	jle	.L634
	movl	$1, %r8d
.L630:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L629:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L629
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L630
	.p2align 4,,10
	.p2align 3
.L633:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L631:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L631
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L633
.L632:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$22>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$22, %r9
	addq	$1408, %rsi
	addq	$1408, %rdx
	cmpq	%r9, %r10
	jg	.L627
.L635:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L634:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L633
.L641:
	leaq	.LC71(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L642:
	leaq	.LC71(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5780:
	.size	_Z15axpb_simd_m_mnmILi22EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi22EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi22EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC72:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 22]"
	.section	.text._Z11axpb_simd_cILi22EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi22EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi22EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi22EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi22EEl14axpb_options_tfPff:
.LFB5779:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$352, 56(%rbp)
	jne	.L659
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L660
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$22>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L651
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L653
	leaq	1408(%rdi), %rdx
.L649:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L648:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L648
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L649
	.p2align 4,,10
	.p2align 3
.L652:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L650:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L650
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L652
	vzeroupper
.L651:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$22>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L653:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	1408(%rdi), %rdx
	jmp	.L652
.L659:
	leaq	.LC72(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L660:
	leaq	.LC72(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5779:
	.size	_Z11axpb_simd_cILi22EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi22EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi23EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC73:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 23]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi23EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi23EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi23EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi23EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi23EEl14axpb_options_tfPff:
.LFB5784:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC73(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5784:
	.size	_Z24axpb_simd_parallel_m_mnmILi23EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi23EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi23EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC74:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 23]"
	.section	.text._Z15axpb_simd_m_mnmILi23EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi23EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi23EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi23EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi23EEl14axpb_options_tfPff:
.LFB5783:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$-5614226457215950491, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	leaq	(%rdx,%rcx), %rax
	movq	%rcx, %rdx
	sarq	$8, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$368, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L683
	cmpq	$1, 32(%rbp)
	jne	.L684
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	1472(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L677
.L669:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$23>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L674
	cmpq	$2, %rdi
	jle	.L676
	movl	$1, %r8d
.L672:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L671:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L671
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L672
	.p2align 4,,10
	.p2align 3
.L675:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L673:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L673
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L675
.L674:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$23>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$23, %r9
	addq	$1472, %rsi
	addq	$1472, %rdx
	cmpq	%r9, %r10
	jg	.L669
.L677:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L676:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L675
.L683:
	leaq	.LC74(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L684:
	leaq	.LC74(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5783:
	.size	_Z15axpb_simd_m_mnmILi23EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi23EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi23EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC75:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 23]"
	.section	.text._Z11axpb_simd_cILi23EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi23EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi23EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi23EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi23EEl14axpb_options_tfPff:
.LFB5782:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$368, 56(%rbp)
	jne	.L701
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L702
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$23>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L693
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L695
	leaq	1472(%rdi), %rdx
.L691:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L690:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L690
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L691
	.p2align 4,,10
	.p2align 3
.L694:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L692:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L692
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L694
	vzeroupper
.L693:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$23>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L695:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	1472(%rdi), %rdx
	jmp	.L694
.L701:
	leaq	.LC75(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L702:
	leaq	.LC75(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5782:
	.size	_Z11axpb_simd_cILi23EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi23EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi24EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC76:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 24]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi24EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi24EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi24EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi24EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi24EEl14axpb_options_tfPff:
.LFB5787:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC76(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5787:
	.size	_Z24axpb_simd_parallel_m_mnmILi24EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi24EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi24EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC77:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 24]"
	.section	.text._Z15axpb_simd_m_mnmILi24EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi24EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi24EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi24EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi24EEl14axpb_options_tfPff:
.LFB5786:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$3074457345618258603, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$6, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	leaq	(%rax,%rax,2), %rax
	salq	$7, %rax
	cmpq	%rax, %rcx
	jne	.L725
	cmpq	$1, 32(%rbp)
	jne	.L726
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	1536(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L719
.L711:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$24>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L716
	cmpq	$2, %rdi
	jle	.L718
	movl	$1, %r8d
.L714:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L713:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L713
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L714
	.p2align 4,,10
	.p2align 3
.L717:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L715:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L715
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L717
.L716:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$24>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$24, %r9
	addq	$1536, %rsi
	addq	$1536, %rdx
	cmpq	%r9, %r10
	jg	.L711
.L719:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L718:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L717
.L725:
	leaq	.LC77(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L726:
	leaq	.LC77(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5786:
	.size	_Z15axpb_simd_m_mnmILi24EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi24EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi24EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC78:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 24]"
	.section	.text._Z11axpb_simd_cILi24EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi24EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi24EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi24EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi24EEl14axpb_options_tfPff:
.LFB5785:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$384, 56(%rbp)
	jne	.L743
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L744
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$24>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L735
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L737
	leaq	1536(%rdi), %rdx
.L733:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L732:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L732
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L733
	.p2align 4,,10
	.p2align 3
.L736:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L734:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L734
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L736
	vzeroupper
.L735:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$24>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L737:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	1536(%rdi), %rdx
	jmp	.L736
.L743:
	leaq	.LC78(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L744:
	leaq	.LC78(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5785:
	.size	_Z11axpb_simd_cILi24EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi24EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi25EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC79:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 25]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi25EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi25EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi25EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi25EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi25EEl14axpb_options_tfPff:
.LFB5790:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC79(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5790:
	.size	_Z24axpb_simd_parallel_m_mnmILi25EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi25EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi25EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC80:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 25]"
	.section	.text._Z15axpb_simd_m_mnmILi25EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi25EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi25EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi25EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi25EEl14axpb_options_tfPff:
.LFB5789:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$-6640827866535438581, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	leaq	(%rdx,%rcx), %rax
	movq	%rcx, %rdx
	sarq	$8, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$400, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L767
	cmpq	$1, 32(%rbp)
	jne	.L768
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	1600(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L761
.L753:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$25>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L758
	cmpq	$2, %rdi
	jle	.L760
	movl	$1, %r8d
.L756:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L755:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L755
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L756
	.p2align 4,,10
	.p2align 3
.L759:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L757:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L757
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L759
.L758:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$25>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$25, %r9
	addq	$1600, %rsi
	addq	$1600, %rdx
	cmpq	%r9, %r10
	jg	.L753
.L761:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L760:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L759
.L767:
	leaq	.LC80(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L768:
	leaq	.LC80(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5789:
	.size	_Z15axpb_simd_m_mnmILi25EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi25EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi25EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC81:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 25]"
	.section	.text._Z11axpb_simd_cILi25EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi25EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi25EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi25EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi25EEl14axpb_options_tfPff:
.LFB5788:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$400, 56(%rbp)
	jne	.L785
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L786
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$25>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L777
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L779
	leaq	1600(%rdi), %rdx
.L775:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L774:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L774
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L775
	.p2align 4,,10
	.p2align 3
.L778:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L776:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L776
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L778
	vzeroupper
.L777:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$25>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L779:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	1600(%rdi), %rdx
	jmp	.L778
.L785:
	leaq	.LC81(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L786:
	leaq	.LC81(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5788:
	.size	_Z11axpb_simd_cILi25EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi25EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi26EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC82:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 26]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi26EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi26EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi26EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi26EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi26EEl14axpb_options_tfPff:
.LFB5793:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC82(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5793:
	.size	_Z24axpb_simd_parallel_m_mnmILi26EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi26EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi26EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC83:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 26]"
	.section	.text._Z15axpb_simd_m_mnmILi26EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi26EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi26EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi26EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi26EEl14axpb_options_tfPff:
.LFB5792:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$5675921253449092805, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$7, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$416, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L809
	cmpq	$1, 32(%rbp)
	jne	.L810
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	1664(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L803
.L795:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$26>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L800
	cmpq	$2, %rdi
	jle	.L802
	movl	$1, %r8d
.L798:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L797:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L797
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L798
	.p2align 4,,10
	.p2align 3
.L801:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L799:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L799
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L801
.L800:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$26>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$26, %r9
	addq	$1664, %rsi
	addq	$1664, %rdx
	cmpq	%r9, %r10
	jg	.L795
.L803:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L802:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L801
.L809:
	leaq	.LC83(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L810:
	leaq	.LC83(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5792:
	.size	_Z15axpb_simd_m_mnmILi26EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi26EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi26EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC84:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 26]"
	.section	.text._Z11axpb_simd_cILi26EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi26EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi26EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi26EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi26EEl14axpb_options_tfPff:
.LFB5791:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$416, 56(%rbp)
	jne	.L827
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L828
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$26>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L819
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L821
	leaq	1664(%rdi), %rdx
.L817:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L816:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L816
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L817
	.p2align 4,,10
	.p2align 3
.L820:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L818:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L818
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L820
	vzeroupper
.L819:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$26>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L821:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	1664(%rdi), %rdx
	jmp	.L820
.L827:
	leaq	.LC84(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L828:
	leaq	.LC84(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5791:
	.size	_Z11axpb_simd_cILi26EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi26EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi27EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC85:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 27]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi27EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi27EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi27EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi27EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi27EEl14axpb_options_tfPff:
.LFB5796:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC85(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5796:
	.size	_Z24axpb_simd_parallel_m_mnmILi27EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi27EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi27EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC86:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 27]"
	.section	.text._Z15axpb_simd_m_mnmILi27EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi27EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi27EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi27EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi27EEl14axpb_options_tfPff:
.LFB5795:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$341606371735362067, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$3, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$432, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L851
	cmpq	$1, 32(%rbp)
	jne	.L852
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	1728(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L845
.L837:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$27>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L842
	cmpq	$2, %rdi
	jle	.L844
	movl	$1, %r8d
.L840:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L839:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L839
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L840
	.p2align 4,,10
	.p2align 3
.L843:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L841:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L841
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L843
.L842:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$27>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$27, %r9
	addq	$1728, %rsi
	addq	$1728, %rdx
	cmpq	%r9, %r10
	jg	.L837
.L845:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L844:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L843
.L851:
	leaq	.LC86(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L852:
	leaq	.LC86(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5795:
	.size	_Z15axpb_simd_m_mnmILi27EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi27EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi27EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC87:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 27]"
	.section	.text._Z11axpb_simd_cILi27EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi27EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi27EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi27EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi27EEl14axpb_options_tfPff:
.LFB5794:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$432, 56(%rbp)
	jne	.L869
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L870
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$27>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L861
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L863
	leaq	1728(%rdi), %rdx
.L859:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L858:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L858
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L859
	.p2align 4,,10
	.p2align 3
.L862:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L860:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L860
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L862
	vzeroupper
.L861:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$27>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L863:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	1728(%rdi), %rdx
	jmp	.L862
.L869:
	leaq	.LC87(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L870:
	leaq	.LC87(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5794:
	.size	_Z11axpb_simd_cILi27EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi27EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi28EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC88:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 28]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi28EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi28EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi28EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi28EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi28EEl14axpb_options_tfPff:
.LFB5799:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC88(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5799:
	.size	_Z24axpb_simd_parallel_m_mnmILi28EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi28EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi28EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC89:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 28]"
	.section	.text._Z15axpb_simd_m_mnmILi28EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi28EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi28EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi28EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi28EEl14axpb_options_tfPff:
.LFB5798:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$5270498306774157605, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$7, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$448, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L893
	cmpq	$1, 32(%rbp)
	jne	.L894
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	1792(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L887
.L879:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$28>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L884
	cmpq	$2, %rdi
	jle	.L886
	movl	$1, %r8d
.L882:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L881:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L881
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L882
	.p2align 4,,10
	.p2align 3
.L885:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L883:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L883
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L885
.L884:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$28>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$28, %r9
	addq	$1792, %rsi
	addq	$1792, %rdx
	cmpq	%r9, %r10
	jg	.L879
.L887:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L886:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L885
.L893:
	leaq	.LC89(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L894:
	leaq	.LC89(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5798:
	.size	_Z15axpb_simd_m_mnmILi28EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi28EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi28EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC90:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 28]"
	.section	.text._Z11axpb_simd_cILi28EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi28EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi28EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi28EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi28EEl14axpb_options_tfPff:
.LFB5797:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$448, 56(%rbp)
	jne	.L911
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L912
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$28>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L903
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L905
	leaq	1792(%rdi), %rdx
.L901:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L900:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L900
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L901
	.p2align 4,,10
	.p2align 3
.L904:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L902:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L902
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L904
	vzeroupper
.L903:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$28>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L905:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	1792(%rdi), %rdx
	jmp	.L904
.L911:
	leaq	.LC90(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L912:
	leaq	.LC90(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5797:
	.size	_Z11axpb_simd_cILi28EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi28EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi29EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC91:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 29]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi29EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi29EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi29EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi29EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi29EEl14axpb_options_tfPff:
.LFB5802:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC91(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5802:
	.size	_Z24axpb_simd_parallel_m_mnmILi29EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi29EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi29EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC92:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 29]"
	.section	.text._Z15axpb_simd_m_mnmILi29EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi29EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi29EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi29EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi29EEl14axpb_options_tfPff:
.LFB5801:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$5088756985850910791, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$7, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$464, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L935
	cmpq	$1, 32(%rbp)
	jne	.L936
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	1856(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L929
.L921:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$29>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L926
	cmpq	$2, %rdi
	jle	.L928
	movl	$1, %r8d
.L924:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L923:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L923
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L924
	.p2align 4,,10
	.p2align 3
.L927:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L925:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L925
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L927
.L926:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$29>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$29, %r9
	addq	$1856, %rsi
	addq	$1856, %rdx
	cmpq	%r9, %r10
	jg	.L921
.L929:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L928:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L927
.L935:
	leaq	.LC92(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L936:
	leaq	.LC92(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5801:
	.size	_Z15axpb_simd_m_mnmILi29EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi29EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi29EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC93:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 29]"
	.section	.text._Z11axpb_simd_cILi29EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi29EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi29EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi29EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi29EEl14axpb_options_tfPff:
.LFB5800:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$464, 56(%rbp)
	jne	.L953
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L954
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$29>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L945
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L947
	leaq	1856(%rdi), %rdx
.L943:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L942:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L942
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L943
	.p2align 4,,10
	.p2align 3
.L946:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L944:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L944
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L946
	vzeroupper
.L945:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$29>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L947:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	1856(%rdi), %rdx
	jmp	.L946
.L953:
	leaq	.LC93(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L954:
	leaq	.LC93(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5800:
	.size	_Z11axpb_simd_cILi29EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi29EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi30EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC94:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 30]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi30EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi30EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi30EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi30EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi30EEl14axpb_options_tfPff:
.LFB5805:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC94(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5805:
	.size	_Z24axpb_simd_parallel_m_mnmILi30EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi30EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi30EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC95:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 30]"
	.section	.text._Z15axpb_simd_m_mnmILi30EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi30EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi30EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi30EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi30EEl14axpb_options_tfPff:
.LFB5804:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$-8608480567731124087, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	leaq	(%rdx,%rcx), %rax
	movq	%rcx, %rdx
	sarq	$8, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$480, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L977
	cmpq	$1, 32(%rbp)
	jne	.L978
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	1920(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L971
.L963:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$30>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L968
	cmpq	$2, %rdi
	jle	.L970
	movl	$1, %r8d
.L966:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L965:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L965
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L966
	.p2align 4,,10
	.p2align 3
.L969:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L967:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L967
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L969
.L968:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$30>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$30, %r9
	addq	$1920, %rsi
	addq	$1920, %rdx
	cmpq	%r9, %r10
	jg	.L963
.L971:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L970:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L969
.L977:
	leaq	.LC95(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L978:
	leaq	.LC95(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5804:
	.size	_Z15axpb_simd_m_mnmILi30EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi30EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi30EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC96:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 30]"
	.section	.text._Z11axpb_simd_cILi30EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi30EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi30EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi30EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi30EEl14axpb_options_tfPff:
.LFB5803:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$480, 56(%rbp)
	jne	.L995
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L996
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$30>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L987
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L989
	leaq	1920(%rdi), %rdx
.L985:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L984:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L984
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L985
	.p2align 4,,10
	.p2align 3
.L988:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L986:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L986
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L988
	vzeroupper
.L987:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$30>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L989:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	1920(%rdi), %rdx
	jmp	.L988
.L995:
	leaq	.LC96(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L996:
	leaq	.LC96(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5803:
	.size	_Z11axpb_simd_cILi30EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi30EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi31EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC97:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 31]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi31EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi31EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi31EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi31EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi31EEl14axpb_options_tfPff:
.LFB5808:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC97(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5808:
	.size	_Z24axpb_simd_parallel_m_mnmILi31EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi31EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi31EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC98:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 31]"
	.section	.text._Z15axpb_simd_m_mnmILi31EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi31EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi31EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi31EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi31EEl14axpb_options_tfPff:
.LFB5807:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$-8925843906633654007, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	leaq	(%rdx,%rcx), %rax
	movq	%rcx, %rdx
	sarq	$8, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$496, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L1019
	cmpq	$1, 32(%rbp)
	jne	.L1020
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	1984(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1013
.L1005:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$31>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1010
	cmpq	$2, %rdi
	jle	.L1012
	movl	$1, %r8d
.L1008:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1007:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1007
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L1008
	.p2align 4,,10
	.p2align 3
.L1011:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1009:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1009
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1011
.L1010:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$31>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$31, %r9
	addq	$1984, %rsi
	addq	$1984, %rdx
	cmpq	%r9, %r10
	jg	.L1005
.L1013:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1012:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1011
.L1019:
	leaq	.LC98(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1020:
	leaq	.LC98(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5807:
	.size	_Z15axpb_simd_m_mnmILi31EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi31EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi31EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC99:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 31]"
	.section	.text._Z11axpb_simd_cILi31EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi31EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi31EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi31EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi31EEl14axpb_options_tfPff:
.LFB5806:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$496, 56(%rbp)
	jne	.L1037
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1038
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$31>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1029
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1031
	leaq	1984(%rdi), %rdx
.L1027:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1026:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1026
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1027
	.p2align 4,,10
	.p2align 3
.L1030:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1028:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1028
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1030
	vzeroupper
.L1029:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$31>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1031:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	1984(%rdi), %rdx
	jmp	.L1030
.L1037:
	leaq	.LC99(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1038:
	leaq	.LC99(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5806:
	.size	_Z11axpb_simd_cILi31EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi31EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi32EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC100:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 32]"
	.section	.text._Z15axpb_simd_m_mnmILi32EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi32EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi32EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi32EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi32EEl14axpb_options_tfPff:
.LFB5810:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rax
	movq	64(%rbp), %rdi
	andq	$-64, %rsp
	testl	$511, %eax
	jne	.L1059
	cmpq	$1, 32(%rbp)
	jne	.L1060
	testq	%rax, %rax
	leaq	15(%rax), %r10
	cmovns	%rax, %r10
	leaq	2048(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rax
	jle	.L1053
.L1045:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$32>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1050
	cmpq	$2, %rdi
	jle	.L1052
	movl	$1, %r8d
.L1048:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1047:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1047
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L1048
	.p2align 4,,10
	.p2align 3
.L1051:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1049:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1049
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1051
.L1050:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$32>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$32, %r9
	addq	$2048, %rsi
	addq	$2048, %rdx
	cmpq	%r9, %r10
	jg	.L1045
.L1053:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1052:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1051
.L1059:
	leaq	.LC100(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1060:
	leaq	.LC100(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5810:
	.size	_Z15axpb_simd_m_mnmILi32EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi32EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi32EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC101:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 32]"
	.section	.text._Z11axpb_simd_cILi32EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi32EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi32EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi32EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi32EEl14axpb_options_tfPff:
.LFB5809:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$512, 56(%rbp)
	jne	.L1077
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1078
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$32>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1069
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1071
	leaq	2048(%rdi), %rdx
.L1067:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1066:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1066
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1067
	.p2align 4,,10
	.p2align 3
.L1070:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1068:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1068
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1070
	vzeroupper
.L1069:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$32>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1071:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	2048(%rdi), %rdx
	jmp	.L1070
.L1077:
	leaq	.LC101(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1078:
	leaq	.LC101(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5809:
	.size	_Z11axpb_simd_cILi32EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi32EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi33EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC102:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 33]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi33EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi33EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi33EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi33EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi33EEl14axpb_options_tfPff:
.LFB5814:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC102(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5814:
	.size	_Z24axpb_simd_parallel_m_mnmILi33EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi33EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi33EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC103:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 33]"
	.section	.text._Z15axpb_simd_m_mnmILi33EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi33EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi33EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi33EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi33EEl14axpb_options_tfPff:
.LFB5813:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$1117984489315730401, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$5, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$528, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L1101
	cmpq	$1, 32(%rbp)
	jne	.L1102
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	2112(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r8d, %r8d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1095
.L1087:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$33>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1092
	cmpq	$2, %rdi
	jle	.L1094
	movl	$1, %r9d
.L1090:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1089:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1089
	leaq	1(%r9), %rcx
	addq	$2, %r9
	cmpq	%r9, %r11
	jg	.L1090
	.p2align 4,,10
	.p2align 3
.L1093:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1091:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1091
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1093
.L1092:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$33>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$33, %r8
	addq	$2112, %rsi
	addq	$2112, %rdx
	cmpq	%r8, %r10
	jg	.L1087
.L1095:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1094:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1093
.L1101:
	leaq	.LC103(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1102:
	leaq	.LC103(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5813:
	.size	_Z15axpb_simd_m_mnmILi33EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi33EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi33EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC104:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 33]"
	.section	.text._Z11axpb_simd_cILi33EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi33EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi33EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi33EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi33EEl14axpb_options_tfPff:
.LFB5812:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$528, 56(%rbp)
	jne	.L1119
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1120
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$33>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1111
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1113
	leaq	2112(%rdi), %rdx
.L1109:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1108:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1108
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1109
	.p2align 4,,10
	.p2align 3
.L1112:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1110:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1110
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1112
	vzeroupper
.L1111:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$33>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1113:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	2112(%rdi), %rdx
	jmp	.L1112
.L1119:
	leaq	.LC104(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1120:
	leaq	.LC104(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5812:
	.size	_Z11axpb_simd_cILi33EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi33EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi34EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC105:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 34]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi34EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi34EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi34EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi34EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi34EEl14axpb_options_tfPff:
.LFB5817:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC105(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5817:
	.size	_Z24axpb_simd_parallel_m_mnmILi34EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi34EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi34EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC106:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 34]"
	.section	.text._Z15axpb_simd_m_mnmILi34EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi34EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi34EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi34EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi34EEl14axpb_options_tfPff:
.LFB5816:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$8680820740569200761, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$8, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$544, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L1143
	cmpq	$1, 32(%rbp)
	jne	.L1144
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	2176(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r8d, %r8d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1137
.L1129:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$34>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1134
	cmpq	$2, %rdi
	jle	.L1136
	movl	$1, %r9d
.L1132:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1131:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1131
	leaq	1(%r9), %rcx
	addq	$2, %r9
	cmpq	%r9, %r11
	jg	.L1132
	.p2align 4,,10
	.p2align 3
.L1135:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1133:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1133
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1135
.L1134:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$34>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$34, %r8
	addq	$2176, %rsi
	addq	$2176, %rdx
	cmpq	%r8, %r10
	jg	.L1129
.L1137:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1136:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1135
.L1143:
	leaq	.LC106(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1144:
	leaq	.LC106(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5816:
	.size	_Z15axpb_simd_m_mnmILi34EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi34EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi34EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC107:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 34]"
	.section	.text._Z11axpb_simd_cILi34EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi34EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi34EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi34EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi34EEl14axpb_options_tfPff:
.LFB5815:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$544, 56(%rbp)
	jne	.L1161
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1162
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$34>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1153
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1155
	leaq	2176(%rdi), %rdx
.L1151:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1150:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1150
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1151
	.p2align 4,,10
	.p2align 3
.L1154:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1152:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1152
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1154
	vzeroupper
.L1153:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$34>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1155:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	2176(%rdi), %rdx
	jmp	.L1154
.L1161:
	leaq	.LC107(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1162:
	leaq	.LC107(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5815:
	.size	_Z11axpb_simd_cILi34EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi34EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi35EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC108:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 35]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi35EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi35EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi35EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi35EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi35EEl14axpb_options_tfPff:
.LFB5820:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC108(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5820:
	.size	_Z24axpb_simd_parallel_m_mnmILi35EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi35EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi35EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC109:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 35]"
	.section	.text._Z15axpb_simd_m_mnmILi35EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi35EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi35EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi35EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi35EEl14axpb_options_tfPff:
.LFB5819:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$1054099661354831521, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$5, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$560, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L1185
	cmpq	$1, 32(%rbp)
	jne	.L1186
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	2240(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r8d, %r8d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1179
.L1171:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$35>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1176
	cmpq	$2, %rdi
	jle	.L1178
	movl	$1, %r9d
.L1174:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1173:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1173
	leaq	1(%r9), %rcx
	addq	$2, %r9
	cmpq	%r9, %r11
	jg	.L1174
	.p2align 4,,10
	.p2align 3
.L1177:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1175:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1175
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1177
.L1176:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$35>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$35, %r8
	addq	$2240, %rsi
	addq	$2240, %rdx
	cmpq	%r8, %r10
	jg	.L1171
.L1179:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1178:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1177
.L1185:
	leaq	.LC109(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1186:
	leaq	.LC109(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5819:
	.size	_Z15axpb_simd_m_mnmILi35EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi35EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi35EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC110:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 35]"
	.section	.text._Z11axpb_simd_cILi35EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi35EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi35EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi35EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi35EEl14axpb_options_tfPff:
.LFB5818:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$560, 56(%rbp)
	jne	.L1203
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1204
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$35>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1195
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1197
	leaq	2240(%rdi), %rdx
.L1193:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1192:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1192
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1193
	.p2align 4,,10
	.p2align 3
.L1196:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1194:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1194
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1196
	vzeroupper
.L1195:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$35>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1197:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	2240(%rdi), %rdx
	jmp	.L1196
.L1203:
	leaq	.LC110(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1204:
	leaq	.LC110(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5818:
	.size	_Z11axpb_simd_cILi35EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi35EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi36EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC111:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 36]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi36EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi36EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi36EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi36EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi36EEl14axpb_options_tfPff:
.LFB5823:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC111(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5823:
	.size	_Z24axpb_simd_parallel_m_mnmILi36EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi36EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi36EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC112:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 36]"
	.section	.text._Z15axpb_simd_m_mnmILi36EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi36EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi36EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi36EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi36EEl14axpb_options_tfPff:
.LFB5822:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$1024819115206086201, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$5, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	leaq	(%rax,%rax,8), %rax
	salq	$6, %rax
	cmpq	%rax, %rcx
	jne	.L1227
	cmpq	$1, 32(%rbp)
	jne	.L1228
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	2304(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r8d, %r8d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1221
.L1213:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$36>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1218
	cmpq	$2, %rdi
	jle	.L1220
	movl	$1, %r9d
.L1216:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1215:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1215
	leaq	1(%r9), %rcx
	addq	$2, %r9
	cmpq	%r9, %r11
	jg	.L1216
	.p2align 4,,10
	.p2align 3
.L1219:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1217:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1217
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1219
.L1218:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$36>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$36, %r8
	addq	$2304, %rsi
	addq	$2304, %rdx
	cmpq	%r8, %r10
	jg	.L1213
.L1221:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1220:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1219
.L1227:
	leaq	.LC112(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1228:
	leaq	.LC112(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5822:
	.size	_Z15axpb_simd_m_mnmILi36EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi36EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi36EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC113:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 36]"
	.section	.text._Z11axpb_simd_cILi36EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi36EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi36EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi36EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi36EEl14axpb_options_tfPff:
.LFB5821:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$576, 56(%rbp)
	jne	.L1245
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1246
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$36>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1237
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1239
	leaq	2304(%rdi), %rdx
.L1235:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1234:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1234
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1235
	.p2align 4,,10
	.p2align 3
.L1238:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1236:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1236
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1238
	vzeroupper
.L1237:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$36>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1239:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	2304(%rdi), %rdx
	jmp	.L1238
.L1245:
	leaq	.LC113(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1246:
	leaq	.LC113(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5821:
	.size	_Z11axpb_simd_cILi36EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi36EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi37EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC114:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 37]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi37EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi37EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi37EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi37EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi37EEl14axpb_options_tfPff:
.LFB5826:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC114(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5826:
	.size	_Z24axpb_simd_parallel_m_mnmILi37EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi37EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi37EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC115:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 37]"
	.section	.text._Z15axpb_simd_m_mnmILi37EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi37EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi37EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi37EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi37EEl14axpb_options_tfPff:
.LFB5825:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$3988485205126389539, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$7, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$592, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L1269
	cmpq	$1, 32(%rbp)
	jne	.L1270
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	2368(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r8d, %r8d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1263
.L1255:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$37>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1260
	cmpq	$2, %rdi
	jle	.L1262
	movl	$1, %r9d
.L1258:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1257:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1257
	leaq	1(%r9), %rcx
	addq	$2, %r9
	cmpq	%r9, %r11
	jg	.L1258
	.p2align 4,,10
	.p2align 3
.L1261:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1259:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1259
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1261
.L1260:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$37>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$37, %r8
	addq	$2368, %rsi
	addq	$2368, %rdx
	cmpq	%r8, %r10
	jg	.L1255
.L1263:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1262:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1261
.L1269:
	leaq	.LC115(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1270:
	leaq	.LC115(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5825:
	.size	_Z15axpb_simd_m_mnmILi37EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi37EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi37EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC116:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 37]"
	.section	.text._Z11axpb_simd_cILi37EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi37EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi37EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi37EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi37EEl14axpb_options_tfPff:
.LFB5824:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$592, 56(%rbp)
	jne	.L1287
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1288
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$37>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1279
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1281
	leaq	2368(%rdi), %rdx
.L1277:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1276:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1276
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1277
	.p2align 4,,10
	.p2align 3
.L1280:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1278:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1278
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1280
	vzeroupper
.L1279:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$37>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1281:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	2368(%rdi), %rdx
	jmp	.L1280
.L1287:
	leaq	.LC116(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1288:
	leaq	.LC116(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5824:
	.size	_Z11axpb_simd_cILi37EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi37EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi38EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC117:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 38]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi38EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi38EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi38EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi38EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi38EEl14axpb_options_tfPff:
.LFB5829:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC117(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5829:
	.size	_Z24axpb_simd_parallel_m_mnmILi38EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi38EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi38EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC118:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 38]"
	.section	.text._Z15axpb_simd_m_mnmILi38EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi38EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi38EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi38EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi38EEl14axpb_options_tfPff:
.LFB5828:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$485440633518672411, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$4, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$608, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L1311
	cmpq	$1, 32(%rbp)
	jne	.L1312
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	2432(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r8d, %r8d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1305
.L1297:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$38>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1302
	cmpq	$2, %rdi
	jle	.L1304
	movl	$1, %r9d
.L1300:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1299:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1299
	leaq	1(%r9), %rcx
	addq	$2, %r9
	cmpq	%r9, %r11
	jg	.L1300
	.p2align 4,,10
	.p2align 3
.L1303:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1301:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1301
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1303
.L1302:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$38>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$38, %r8
	addq	$2432, %rsi
	addq	$2432, %rdx
	cmpq	%r8, %r10
	jg	.L1297
.L1305:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1304:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1303
.L1311:
	leaq	.LC118(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1312:
	leaq	.LC118(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5828:
	.size	_Z15axpb_simd_m_mnmILi38EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi38EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi38EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC119:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 38]"
	.section	.text._Z11axpb_simd_cILi38EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi38EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi38EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi38EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi38EEl14axpb_options_tfPff:
.LFB5827:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$608, 56(%rbp)
	jne	.L1329
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1330
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$38>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1321
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1323
	leaq	2432(%rdi), %rdx
.L1319:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1318:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1318
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1319
	.p2align 4,,10
	.p2align 3
.L1322:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1320:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1320
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1322
	vzeroupper
.L1321:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$38>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1323:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	2432(%rdi), %rdx
	jmp	.L1322
.L1329:
	leaq	.LC119(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1330:
	leaq	.LC119(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5827:
	.size	_Z11axpb_simd_cILi38EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi38EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi39EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC120:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 39]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi39EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi39EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi39EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi39EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi39EEl14axpb_options_tfPff:
.LFB5832:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC120(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5832:
	.size	_Z24axpb_simd_parallel_m_mnmILi39EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi39EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi39EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC121:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 39]"
	.section	.text._Z15axpb_simd_m_mnmILi39EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi39EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi39EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi39EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi39EEl14axpb_options_tfPff:
.LFB5831:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$7567895004598790407, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$8, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$624, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L1353
	cmpq	$1, 32(%rbp)
	jne	.L1354
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	2496(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r8d, %r8d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1347
.L1339:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$39>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1344
	cmpq	$2, %rdi
	jle	.L1346
	movl	$1, %r9d
.L1342:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1341:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1341
	leaq	1(%r9), %rcx
	addq	$2, %r9
	cmpq	%r9, %r11
	jg	.L1342
	.p2align 4,,10
	.p2align 3
.L1345:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1343:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1343
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1345
.L1344:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$39>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$39, %r8
	addq	$2496, %rsi
	addq	$2496, %rdx
	cmpq	%r8, %r10
	jg	.L1339
.L1347:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1346:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1345
.L1353:
	leaq	.LC121(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1354:
	leaq	.LC121(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5831:
	.size	_Z15axpb_simd_m_mnmILi39EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi39EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi39EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC122:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 39]"
	.section	.text._Z11axpb_simd_cILi39EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi39EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi39EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi39EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi39EEl14axpb_options_tfPff:
.LFB5830:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$624, 56(%rbp)
	jne	.L1371
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1372
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$39>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1363
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1365
	leaq	2496(%rdi), %rdx
.L1361:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1360:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1360
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1361
	.p2align 4,,10
	.p2align 3
.L1364:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1362:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1362
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1364
	vzeroupper
.L1363:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$39>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1365:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	2496(%rdi), %rdx
	jmp	.L1364
.L1371:
	leaq	.LC122(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1372:
	leaq	.LC122(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5830:
	.size	_Z11axpb_simd_cILi39EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi39EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi40EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC123:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 40]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi40EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi40EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi40EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi40EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi40EEl14axpb_options_tfPff:
.LFB5835:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC123(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5835:
	.size	_Z24axpb_simd_parallel_m_mnmILi40EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi40EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi40EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC124:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 40]"
	.section	.text._Z15axpb_simd_m_mnmILi40EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi40EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi40EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi40EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi40EEl14axpb_options_tfPff:
.LFB5834:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$7378697629483820647, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$8, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	leaq	(%rax,%rax,4), %rax
	salq	$7, %rax
	cmpq	%rax, %rcx
	jne	.L1395
	cmpq	$1, 32(%rbp)
	jne	.L1396
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	2560(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r8d, %r8d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1389
.L1381:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$40>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1386
	cmpq	$2, %rdi
	jle	.L1388
	movl	$1, %r9d
.L1384:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1383:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1383
	leaq	1(%r9), %rcx
	addq	$2, %r9
	cmpq	%r9, %r11
	jg	.L1384
	.p2align 4,,10
	.p2align 3
.L1387:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1385:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1385
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1387
.L1386:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$40>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$40, %r8
	addq	$2560, %rsi
	addq	$2560, %rdx
	cmpq	%r8, %r10
	jg	.L1381
.L1389:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1388:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1387
.L1395:
	leaq	.LC124(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1396:
	leaq	.LC124(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5834:
	.size	_Z15axpb_simd_m_mnmILi40EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi40EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi40EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC125:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 40]"
	.section	.text._Z11axpb_simd_cILi40EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi40EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi40EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi40EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi40EEl14axpb_options_tfPff:
.LFB5833:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$640, 56(%rbp)
	jne	.L1413
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1414
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$40>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1405
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1407
	leaq	2560(%rdi), %rdx
.L1403:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1402:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1402
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1403
	.p2align 4,,10
	.p2align 3
.L1406:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1404:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1404
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1406
	vzeroupper
.L1405:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$40>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1407:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	2560(%rdi), %rdx
	jmp	.L1406
.L1413:
	leaq	.LC125(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1414:
	leaq	.LC125(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5833:
	.size	_Z11axpb_simd_cILi40EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi40EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi41EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC126:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 41]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi41EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi41EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi41EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi41EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi41EEl14axpb_options_tfPff:
.LFB5838:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC126(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5838:
	.size	_Z24axpb_simd_parallel_m_mnmILi41EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi41EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi41EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC127:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 41]"
	.section	.text._Z15axpb_simd_m_mnmILi41EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi41EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi41EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi41EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi41EEl14axpb_options_tfPff:
.LFB5837:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$7198729394618361607, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$8, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$656, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L1437
	cmpq	$1, 32(%rbp)
	jne	.L1438
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	2624(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1431
.L1423:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$41>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1428
	cmpq	$2, %rdi
	jle	.L1430
	movl	$1, %r8d
.L1426:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1425:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1425
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L1426
	.p2align 4,,10
	.p2align 3
.L1429:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1427:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1427
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1429
.L1428:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$41>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$41, %r9
	addq	$2624, %rsi
	addq	$2624, %rdx
	cmpq	%r9, %r10
	jg	.L1423
.L1431:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1430:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1429
.L1437:
	leaq	.LC127(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1438:
	leaq	.LC127(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5837:
	.size	_Z15axpb_simd_m_mnmILi41EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi41EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi41EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC128:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 41]"
	.section	.text._Z11axpb_simd_cILi41EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi41EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi41EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi41EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi41EEl14axpb_options_tfPff:
.LFB5836:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$656, 56(%rbp)
	jne	.L1455
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1456
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$41>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1447
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1449
	leaq	2624(%rdi), %rdx
.L1445:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1444:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1444
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1445
	.p2align 4,,10
	.p2align 3
.L1448:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1446:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1446
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1448
	vzeroupper
.L1447:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$41>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1449:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	2624(%rdi), %rdx
	jmp	.L1448
.L1455:
	leaq	.LC128(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1456:
	leaq	.LC128(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5836:
	.size	_Z11axpb_simd_cILi41EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi41EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi42EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC129:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 42]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi42EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi42EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi42EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi42EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi42EEl14axpb_options_tfPff:
.LFB5841:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC129(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5841:
	.size	_Z24axpb_simd_parallel_m_mnmILi42EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi42EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi42EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC130:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 42]"
	.section	.text._Z15axpb_simd_m_mnmILi42EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi42EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi42EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi42EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi42EEl14axpb_options_tfPff:
.LFB5840:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$-4392081922311798003, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	leaq	(%rdx,%rcx), %rax
	movq	%rcx, %rdx
	sarq	$9, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$672, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L1479
	cmpq	$1, 32(%rbp)
	jne	.L1480
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	2688(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1473
.L1465:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$42>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1470
	cmpq	$2, %rdi
	jle	.L1472
	movl	$1, %r8d
.L1468:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1467:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1467
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L1468
	.p2align 4,,10
	.p2align 3
.L1471:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1469:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1469
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1471
.L1470:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$42>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$42, %r9
	addq	$2688, %rsi
	addq	$2688, %rdx
	cmpq	%r9, %r10
	jg	.L1465
.L1473:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1472:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1471
.L1479:
	leaq	.LC130(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1480:
	leaq	.LC130(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5840:
	.size	_Z15axpb_simd_m_mnmILi42EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi42EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi42EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC131:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 42]"
	.section	.text._Z11axpb_simd_cILi42EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi42EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi42EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi42EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi42EEl14axpb_options_tfPff:
.LFB5839:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$672, 56(%rbp)
	jne	.L1497
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1498
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$42>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1489
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1491
	leaq	2688(%rdi), %rdx
.L1487:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1486:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1486
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1487
	.p2align 4,,10
	.p2align 3
.L1490:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1488:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1488
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1490
	vzeroupper
.L1489:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$42>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1491:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	2688(%rdi), %rdx
	jmp	.L1490
.L1497:
	leaq	.LC131(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1498:
	leaq	.LC131(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5839:
	.size	_Z11axpb_simd_cILi42EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi42EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi43EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC132:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 43]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi43EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi43EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi43EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi43EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi43EEl14axpb_options_tfPff:
.LFB5844:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC132(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5844:
	.size	_Z24axpb_simd_parallel_m_mnmILi43EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi43EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi43EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC133:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 43]"
	.section	.text._Z15axpb_simd_m_mnmILi43EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi43EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi43EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi43EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi43EEl14axpb_options_tfPff:
.LFB5843:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$214497024112901763, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$3, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$688, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L1521
	cmpq	$1, 32(%rbp)
	jne	.L1522
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	2752(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1515
.L1507:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$43>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1512
	cmpq	$2, %rdi
	jle	.L1514
	movl	$1, %r8d
.L1510:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1509:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1509
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L1510
	.p2align 4,,10
	.p2align 3
.L1513:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1511:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1511
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1513
.L1512:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$43>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$43, %r9
	addq	$2752, %rsi
	addq	$2752, %rdx
	cmpq	%r9, %r10
	jg	.L1507
.L1515:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1514:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1513
.L1521:
	leaq	.LC133(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1522:
	leaq	.LC133(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5843:
	.size	_Z15axpb_simd_m_mnmILi43EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi43EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi43EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC134:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 43]"
	.section	.text._Z11axpb_simd_cILi43EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi43EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi43EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi43EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi43EEl14axpb_options_tfPff:
.LFB5842:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$688, 56(%rbp)
	jne	.L1539
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1540
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$43>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1531
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1533
	leaq	2752(%rdi), %rdx
.L1529:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1528:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1528
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1529
	.p2align 4,,10
	.p2align 3
.L1532:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1530:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1530
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1532
	vzeroupper
.L1531:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$43>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1533:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	2752(%rdi), %rdx
	jmp	.L1532
.L1539:
	leaq	.LC134(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1540:
	leaq	.LC134(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5842:
	.size	_Z11axpb_simd_cILi43EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi43EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi44EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC135:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 44]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi44EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi44EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi44EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi44EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi44EEl14axpb_options_tfPff:
.LFB5847:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC135(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5847:
	.size	_Z24axpb_simd_parallel_m_mnmILi44EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi44EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi44EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC136:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 44]"
	.section	.text._Z15axpb_simd_m_mnmILi44EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi44EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi44EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi44EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi44EEl14axpb_options_tfPff:
.LFB5846:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$3353953467947191203, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$7, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$704, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L1563
	cmpq	$1, 32(%rbp)
	jne	.L1564
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	2816(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1557
.L1549:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$44>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1554
	cmpq	$2, %rdi
	jle	.L1556
	movl	$1, %r8d
.L1552:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1551:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1551
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L1552
	.p2align 4,,10
	.p2align 3
.L1555:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1553:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1553
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1555
.L1554:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$44>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$44, %r9
	addq	$2816, %rsi
	addq	$2816, %rdx
	cmpq	%r9, %r10
	jg	.L1549
.L1557:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1556:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1555
.L1563:
	leaq	.LC136(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1564:
	leaq	.LC136(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5846:
	.size	_Z15axpb_simd_m_mnmILi44EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi44EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi44EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC137:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 44]"
	.section	.text._Z11axpb_simd_cILi44EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi44EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi44EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi44EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi44EEl14axpb_options_tfPff:
.LFB5845:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$704, 56(%rbp)
	jne	.L1581
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1582
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$44>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1573
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1575
	leaq	2816(%rdi), %rdx
.L1571:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1570:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1570
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1571
	.p2align 4,,10
	.p2align 3
.L1574:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1572:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1572
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1574
	vzeroupper
.L1573:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$44>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1575:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	2816(%rdi), %rdx
	jmp	.L1574
.L1581:
	leaq	.LC137(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1582:
	leaq	.LC137(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5845:
	.size	_Z11axpb_simd_cILi44EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi44EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi45EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC138:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 45]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi45EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi45EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi45EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi45EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi45EEl14axpb_options_tfPff:
.LFB5850:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC138(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5850:
	.size	_Z24axpb_simd_parallel_m_mnmILi45EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi45EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi45EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC139:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 45]"
	.section	.text._Z15axpb_simd_m_mnmILi45EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi45EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi45EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi45EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi45EEl14axpb_options_tfPff:
.LFB5849:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$3279421168659475843, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$7, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$720, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L1605
	cmpq	$1, 32(%rbp)
	jne	.L1606
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	2880(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1599
.L1591:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$45>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1596
	cmpq	$2, %rdi
	jle	.L1598
	movl	$1, %r8d
.L1594:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1593:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1593
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L1594
	.p2align 4,,10
	.p2align 3
.L1597:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1595:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1595
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1597
.L1596:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$45>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$45, %r9
	addq	$2880, %rsi
	addq	$2880, %rdx
	cmpq	%r9, %r10
	jg	.L1591
.L1599:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1598:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1597
.L1605:
	leaq	.LC139(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1606:
	leaq	.LC139(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5849:
	.size	_Z15axpb_simd_m_mnmILi45EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi45EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi45EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC140:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 45]"
	.section	.text._Z11axpb_simd_cILi45EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi45EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi45EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi45EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi45EEl14axpb_options_tfPff:
.LFB5848:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$720, 56(%rbp)
	jne	.L1623
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1624
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$45>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1615
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1617
	leaq	2880(%rdi), %rdx
.L1613:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1612:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1612
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1613
	.p2align 4,,10
	.p2align 3
.L1616:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1614:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1614
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1616
	vzeroupper
.L1615:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$45>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1617:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	2880(%rdi), %rdx
	jmp	.L1616
.L1623:
	leaq	.LC140(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1624:
	leaq	.LC140(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5848:
	.size	_Z11axpb_simd_cILi45EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi45EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi46EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC141:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 46]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi46EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi46EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi46EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi46EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi46EEl14axpb_options_tfPff:
.LFB5853:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC141(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5853:
	.size	_Z24axpb_simd_parallel_m_mnmILi46EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi46EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi46EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC142:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 46]"
	.section	.text._Z15axpb_simd_m_mnmILi46EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi46EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi46EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi46EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi46EEl14axpb_options_tfPff:
.LFB5852:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$-5614226457215950491, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	leaq	(%rdx,%rcx), %rax
	movq	%rcx, %rdx
	sarq	$9, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$736, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L1647
	cmpq	$1, 32(%rbp)
	jne	.L1648
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	2944(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1641
.L1633:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$46>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1638
	cmpq	$2, %rdi
	jle	.L1640
	movl	$1, %r8d
.L1636:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1635:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1635
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L1636
	.p2align 4,,10
	.p2align 3
.L1639:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1637:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1637
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1639
.L1638:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$46>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$46, %r9
	addq	$2944, %rsi
	addq	$2944, %rdx
	cmpq	%r9, %r10
	jg	.L1633
.L1641:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1640:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1639
.L1647:
	leaq	.LC142(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1648:
	leaq	.LC142(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5852:
	.size	_Z15axpb_simd_m_mnmILi46EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi46EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi46EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC143:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 46]"
	.section	.text._Z11axpb_simd_cILi46EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi46EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi46EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi46EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi46EEl14axpb_options_tfPff:
.LFB5851:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$736, 56(%rbp)
	jne	.L1665
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1666
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$46>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1657
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1659
	leaq	2944(%rdi), %rdx
.L1655:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1654:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1654
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1655
	.p2align 4,,10
	.p2align 3
.L1658:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1656:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1656
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1658
	vzeroupper
.L1657:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$46>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1659:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	2944(%rdi), %rdx
	jmp	.L1658
.L1665:
	leaq	.LC143(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1666:
	leaq	.LC143(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5851:
	.size	_Z11axpb_simd_cILi46EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi46EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi47EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC144:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 47]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi47EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi47EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi47EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi47EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi47EEl14axpb_options_tfPff:
.LFB5856:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC144(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5856:
	.size	_Z24axpb_simd_parallel_m_mnmILi47EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi47EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi47EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC145:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 47]"
	.section	.text._Z15axpb_simd_m_mnmILi47EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi47EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi47EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi47EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi47EEl14axpb_options_tfPff:
.LFB5855:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$6279742663390485657, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$8, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$752, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L1689
	cmpq	$1, 32(%rbp)
	jne	.L1690
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	3008(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1683
.L1675:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$47>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1680
	cmpq	$2, %rdi
	jle	.L1682
	movl	$1, %r8d
.L1678:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1677:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1677
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L1678
	.p2align 4,,10
	.p2align 3
.L1681:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1679:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1679
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1681
.L1680:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$47>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$47, %r9
	addq	$3008, %rsi
	addq	$3008, %rdx
	cmpq	%r9, %r10
	jg	.L1675
.L1683:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1682:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1681
.L1689:
	leaq	.LC145(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1690:
	leaq	.LC145(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5855:
	.size	_Z15axpb_simd_m_mnmILi47EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi47EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi47EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC146:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 47]"
	.section	.text._Z11axpb_simd_cILi47EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi47EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi47EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi47EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi47EEl14axpb_options_tfPff:
.LFB5854:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$752, 56(%rbp)
	jne	.L1707
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1708
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$47>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1699
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1701
	leaq	3008(%rdi), %rdx
.L1697:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1696:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1696
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1697
	.p2align 4,,10
	.p2align 3
.L1700:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1698:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1698
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1700
	vzeroupper
.L1699:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$47>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1701:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	3008(%rdi), %rdx
	jmp	.L1700
.L1707:
	leaq	.LC146(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1708:
	leaq	.LC146(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5854:
	.size	_Z11axpb_simd_cILi47EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi47EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi48EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC147:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 48]"
	.section	.text._Z15axpb_simd_m_mnmILi48EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi48EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi48EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi48EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi48EEl14axpb_options_tfPff:
.LFB5858:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$3074457345618258603, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$7, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	leaq	(%rax,%rax,2), %rax
	salq	$8, %rax
	cmpq	%rax, %rcx
	jne	.L1729
	cmpq	$1, 32(%rbp)
	jne	.L1730
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	3072(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1723
.L1715:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$48>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1720
	cmpq	$2, %rdi
	jle	.L1722
	movl	$1, %r8d
.L1718:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1717:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1717
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L1718
	.p2align 4,,10
	.p2align 3
.L1721:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1719:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1719
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1721
.L1720:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$48>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$48, %r9
	addq	$3072, %rsi
	addq	$3072, %rdx
	cmpq	%r9, %r10
	jg	.L1715
.L1723:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1722:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1721
.L1729:
	leaq	.LC147(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1730:
	leaq	.LC147(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5858:
	.size	_Z15axpb_simd_m_mnmILi48EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi48EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi48EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC148:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 48]"
	.section	.text._Z11axpb_simd_cILi48EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi48EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi48EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi48EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi48EEl14axpb_options_tfPff:
.LFB5857:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$768, 56(%rbp)
	jne	.L1747
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1748
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$48>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1739
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1741
	leaq	3072(%rdi), %rdx
.L1737:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1736:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1736
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1737
	.p2align 4,,10
	.p2align 3
.L1740:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1738:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1738
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1740
	vzeroupper
.L1739:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$48>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1741:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	3072(%rdi), %rdx
	jmp	.L1740
.L1747:
	leaq	.LC148(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1748:
	leaq	.LC148(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5857:
	.size	_Z11axpb_simd_cILi48EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi48EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi49EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC149:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 49]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi49EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi49EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi49EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi49EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi49EEl14axpb_options_tfPff:
.LFB5862:
	.cfi_startproc
	endbr64
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	.cfi_def_cfa_offset 8
	leaq	.LC149(%rip), %rcx
	movl	$287, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC6(%rip), %rdi
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5862:
	.size	_Z24axpb_simd_parallel_m_mnmILi49EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi49EEl14axpb_options_tfPff
	.section	.rodata._Z15axpb_simd_m_mnmILi49EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC150:
	.string	"long int axpb_simd_m_mnm(axpb_options_t, float, float*, float) [with int c = 49]"
	.section	.text._Z15axpb_simd_m_mnmILi49EEl14axpb_options_tfPff,"axG",@progbits,_Z15axpb_simd_m_mnmILi49EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z15axpb_simd_m_mnmILi49EEl14axpb_options_tfPff
	.type	_Z15axpb_simd_m_mnmILi49EEl14axpb_options_tfPff, @function
_Z15axpb_simd_m_mnmILi49EEl14axpb_options_tfPff:
.LFB5861:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movabsq	$6023426636313322977, %rdx
	movq	%rdi, %rsi
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rcx
	movq	64(%rbp), %rdi
	movq	%rcx, %rax
	imulq	%rdx
	andq	$-64, %rsp
	movq	%rdx, %rax
	movq	%rcx, %rdx
	sarq	$8, %rax
	sarq	$63, %rdx
	subq	%rdx, %rax
	imulq	$784, %rax, %rax
	cmpq	%rax, %rcx
	jne	.L1771
	cmpq	$1, 32(%rbp)
	jne	.L1772
	testq	%rcx, %rcx
	leaq	15(%rcx), %r10
	cmovns	%rcx, %r10
	leaq	3136(%rsi), %rdx
	sarq	$4, %r10
	xorl	%r9d, %r9d
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rdi), %r11
	cmpq	$15, %rcx
	jle	.L1765
.L1757:
#APP
# 255 "axpb.cc" 1
	# axpb_simd_m_mnm<$49>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rdi, %rdi
	jle	.L1762
	cmpq	$2, %rdi
	jle	.L1764
	movl	$1, %r8d
.L1760:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1759:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1759
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r11
	jg	.L1760
	.p2align 4,,10
	.p2align 3
.L1763:
	movq	%rsi, %rax
	.p2align 4,,10
	.p2align 3
.L1761:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1761
	incq	%rcx
	cmpq	%rcx, %rdi
	jg	.L1763
.L1762:
#APP
# 261 "axpb.cc" 1
	# axpb_simd_m_mnm<$49>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$49, %r9
	addq	$3136, %rsi
	addq	$3136, %rdx
	cmpq	%r9, %r10
	jg	.L1757
.L1765:
	xorl	%eax, %eax
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1764:
	.cfi_restore_state
	xorl	%ecx, %ecx
	jmp	.L1763
.L1771:
	leaq	.LC150(%rip), %rcx
	movl	$251, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC8(%rip), %rdi
	call	__assert_fail@PLT
.L1772:
	leaq	.LC150(%rip), %rcx
	movl	$252, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5861:
	.size	_Z15axpb_simd_m_mnmILi49EEl14axpb_options_tfPff, .-_Z15axpb_simd_m_mnmILi49EEl14axpb_options_tfPff
	.section	.rodata._Z11axpb_simd_cILi49EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC151:
	.string	"long int axpb_simd_c(axpb_options_t, float, float*, float) [with int c = 49]"
	.section	.text._Z11axpb_simd_cILi49EEl14axpb_options_tfPff,"axG",@progbits,_Z11axpb_simd_cILi49EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z11axpb_simd_cILi49EEl14axpb_options_tfPff
	.type	_Z11axpb_simd_cILi49EEl14axpb_options_tfPff, @function
_Z11axpb_simd_cILi49EEl14axpb_options_tfPff:
.LFB5860:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$784, 56(%rbp)
	jne	.L1789
	movq	32(%rbp), %r8
	cmpq	$1, %r8
	jne	.L1790
	movq	64(%rbp), %rsi
#APP
# 153 "axpb.cc" 1
	# axpb_simd_c<$49>: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%rsi, %rsi
	jle	.L1781
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	leaq	-1(%rsi), %r9
	cmpq	$2, %rsi
	jle	.L1783
	leaq	3136(%rdi), %rdx
.L1779:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1778:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1778
	leaq	1(%r8), %rcx
	addq	$2, %r8
	cmpq	%r8, %r9
	jg	.L1779
	.p2align 4,,10
	.p2align 3
.L1782:
	movq	%rdi, %rax
	.p2align 4,,10
	.p2align 3
.L1780:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rdx, %rax
	jne	.L1780
	incq	%rcx
	cmpq	%rcx, %rsi
	jg	.L1782
	vzeroupper
.L1781:
#APP
# 159 "axpb.cc" 1
	# axpb_simd_c<$49>: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1783:
	.cfi_restore_state
	xorl	%ecx, %ecx
	leaq	3136(%rdi), %rdx
	jmp	.L1782
.L1789:
	leaq	.LC151(%rip), %rcx
	movl	$149, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC10(%rip), %rdi
	call	__assert_fail@PLT
.L1790:
	leaq	.LC151(%rip), %rcx
	movl	$150, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5860:
	.size	_Z11axpb_simd_cILi49EEl14axpb_options_tfPff, .-_Z11axpb_simd_cILi49EEl14axpb_options_tfPff
	.section	.rodata.str1.1
.LC152:
	.string	"scalar"
	.section	.rodata.str1.8
	.align 8
.LC153:
	.ascii	"usage:\n\n  %s [options ...]\n\noptions:\n  --help          "
	.ascii	"        show this help\n  -a,--algo A             use algori"
	.ascii	"thm A (scalar,simd,simd_c,simd_m,simd_mnm,simd_nmn,simd_para"
	.ascii	"llel_mnm,cuda) [%s]\n  -b,--cuda-block-size N  set cuda bloc"
	.ascii	"k size to N [%ld]\n  -w,--active-threader-per-warp N  set ac"
	.ascii	"tive t"
	.string	"hreads per warp to N [%ld]\n  -c,--concurrent-vars N  concurrently update N floats [%ld]\n  -m,--vars N             update N floats [%ld]\n  -n,--n N                update each float variable N times [%ld]\n  -s,--seed N             set random seed to N [%ld]\n"
	.text
	.p2align 4
	.type	_ZL5usagePKc, @function
_ZL5usagePKc:
.LFB5663:
	.cfi_startproc
	subq	$16, %rsp
	.cfi_def_cfa_offset 24
	movabsq	$76843802738543, %rax
	pushq	%rax
	.cfi_def_cfa_offset 32
	movq	%rdi, %rcx
	movq	stderr(%rip), %rdi
	pushq	$1000000
	.cfi_def_cfa_offset 40
	movl	$1, %r9d
	leaq	.LC152(%rip), %r8
	pushq	$1
	.cfi_def_cfa_offset 48
	leaq	.LC153(%rip), %rdx
	movl	$1, %esi
	pushq	$1
	.cfi_def_cfa_offset 56
	xorl	%eax, %eax
	pushq	$32
	.cfi_def_cfa_offset 64
	call	__fprintf_chk@PLT
	addq	$56, %rsp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE5663:
	.size	_ZL5usagePKc, .-_ZL5usagePKc
	.section	.text._Z24axpb_simd_parallel_m_mnmILi16EEl14axpb_options_tfPff._omp_fn.0,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi16EEl14axpb_options_tfPff,comdat
	.p2align 4
	.type	_Z24axpb_simd_parallel_m_mnmILi16EEl14axpb_options_tfPff._omp_fn.0, @function
_Z24axpb_simd_parallel_m_mnmILi16EEl14axpb_options_tfPff._omp_fn.0:
.LFB5880:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	pushq	%rbx
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	movq	%rdi, %rbx
	andq	$-64, %rsp
	call	omp_get_num_threads@PLT
	movslq	%eax, %r12
	call	omp_get_thread_num@PLT
	movslq	%eax, %rcx
	movq	(%rbx), %rax
	testq	%rax, %rax
	leaq	15(%rax), %rdx
	cmovns	%rax, %rdx
	sarq	$4, %rdx
	leaq	30(%rdx), %rax
	addq	$15, %rdx
	cmovns	%rdx, %rax
	sarq	$4, %rax
	cqto
	idivq	%r12
	cmpq	%rdx, %rcx
	jl	.L1794
.L1800:
	imulq	%rax, %rcx
	addq	%rcx, %rdx
	addq	%rdx, %rax
	cmpq	%rax, %rdx
	jge	.L1803
	salq	$4, %rax
	movq	%rax, %rsi
	movq	%rdx, %rax
	movq	%rdx, %rdi
	salq	$10, %rax
	movq	8(%rbx), %rcx
	vbroadcastss	24(%rbx), %zmm1
	vbroadcastss	28(%rbx), %zmm0
	salq	$4, %rdi
	addq	16(%rbx), %rax
.L1798:
	vmovups	(%rax), %zmm17
	vmovups	64(%rax), %zmm16
	vmovups	128(%rax), %zmm15
	vmovups	192(%rax), %zmm14
	vmovups	256(%rax), %zmm13
	vmovups	320(%rax), %zmm12
	vmovups	384(%rax), %zmm11
	vmovups	448(%rax), %zmm10
	vmovups	512(%rax), %zmm9
	vmovups	576(%rax), %zmm8
	vmovups	640(%rax), %zmm7
	vmovups	704(%rax), %zmm6
	vmovups	768(%rax), %zmm5
	vmovups	832(%rax), %zmm4
	vmovups	896(%rax), %zmm3
	vmovups	960(%rax), %zmm2
#APP
# 297 "axpb.cc" 1
	# axpb_simd_parallel_m_mnm<$16>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%rcx, %rcx
	jle	.L1796
	xorl	%edx, %edx
	.p2align 4,,10
	.p2align 3
.L1797:
	incq	%rdx
	vfmadd132ps	%zmm1, %zmm0, %zmm17
	vfmadd132ps	%zmm1, %zmm0, %zmm16
	vfmadd132ps	%zmm1, %zmm0, %zmm15
	vfmadd132ps	%zmm1, %zmm0, %zmm14
	vfmadd132ps	%zmm1, %zmm0, %zmm13
	vfmadd132ps	%zmm1, %zmm0, %zmm12
	vfmadd132ps	%zmm1, %zmm0, %zmm11
	vfmadd132ps	%zmm1, %zmm0, %zmm10
	vfmadd132ps	%zmm1, %zmm0, %zmm9
	vfmadd132ps	%zmm1, %zmm0, %zmm8
	vfmadd132ps	%zmm1, %zmm0, %zmm7
	vfmadd132ps	%zmm1, %zmm0, %zmm6
	vfmadd132ps	%zmm1, %zmm0, %zmm5
	vfmadd132ps	%zmm1, %zmm0, %zmm4
	vfmadd132ps	%zmm1, %zmm0, %zmm3
	vfmadd132ps	%zmm1, %zmm0, %zmm2
	cmpq	%rdx, %rcx
	jne	.L1797
.L1796:
#APP
# 303 "axpb.cc" 1
	# axpb_simd_parallel_m_mnm<$16>: ax+c inner loop end
# 0 "" 2
#NO_APP
	addq	$16, %rdi
	vmovups	%zmm17, (%rax)
	vmovups	%zmm16, 64(%rax)
	vmovups	%zmm15, 128(%rax)
	vmovups	%zmm14, 192(%rax)
	vmovups	%zmm13, 256(%rax)
	vmovups	%zmm12, 320(%rax)
	vmovups	%zmm11, 384(%rax)
	vmovups	%zmm10, 448(%rax)
	vmovups	%zmm9, 512(%rax)
	vmovups	%zmm8, 576(%rax)
	vmovups	%zmm7, 640(%rax)
	vmovups	%zmm6, 704(%rax)
	vmovups	%zmm5, 768(%rax)
	vmovups	%zmm4, 832(%rax)
	vmovups	%zmm3, 896(%rax)
	vmovups	%zmm2, 960(%rax)
	addq	$1024, %rax
	cmpq	%rdi, %rsi
	jg	.L1798
	vzeroupper
.L1803:
	leaq	-16(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1794:
	.cfi_restore_state
	incq	%rax
	xorl	%edx, %edx
	jmp	.L1800
	.cfi_endproc
.LFE5880:
	.size	_Z24axpb_simd_parallel_m_mnmILi16EEl14axpb_options_tfPff._omp_fn.0, .-_Z24axpb_simd_parallel_m_mnmILi16EEl14axpb_options_tfPff._omp_fn.0
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi16EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC154:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 16]"
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi16EEl14axpb_options_tfPff.str1.1,"aMS",@progbits,1
.LC155:
	.string	"m % c == 0"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi16EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi16EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi16EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi16EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi16EEl14axpb_options_tfPff:
.LFB5763:
	.cfi_startproc
	endbr64
	subq	$56, %rsp
	.cfi_def_cfa_offset 64
	movq	%fs:40, %rax
	movq	%rax, 40(%rsp)
	xorl	%eax, %eax
	movq	104(%rsp), %rax
	movq	112(%rsp), %rdx
	testb	$15, %al
	jne	.L1810
	cmpq	$1, 80(%rsp)
	jne	.L1811
	movq	%rdi, 16(%rsp)
	movq	%rdx, 8(%rsp)
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	movq	%rsp, %rsi
	leaq	_Z24axpb_simd_parallel_m_mnmILi16EEl14axpb_options_tfPff._omp_fn.0(%rip), %rdi
	movq	%rax, (%rsp)
	vmovss	%xmm1, 28(%rsp)
	vmovss	%xmm0, 24(%rsp)
	call	GOMP_parallel@PLT
	movq	40(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L1812
	xorl	%eax, %eax
	addq	$56, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L1810:
	.cfi_restore_state
	leaq	.LC154(%rip), %rcx
	movl	$288, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC155(%rip), %rdi
	call	__assert_fail@PLT
.L1812:
	call	__stack_chk_fail@PLT
.L1811:
	leaq	.LC154(%rip), %rcx
	movl	$289, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5763:
	.size	_Z24axpb_simd_parallel_m_mnmILi16EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi16EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi32EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC156:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 32]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi32EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi32EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi32EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi32EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi32EEl14axpb_options_tfPff:
.LFB5811:
	.cfi_startproc
	endbr64
	subq	$56, %rsp
	.cfi_def_cfa_offset 64
	movq	%fs:40, %rax
	movq	%rax, 40(%rsp)
	xorl	%eax, %eax
	movq	104(%rsp), %rax
	movq	112(%rsp), %rdx
	testb	$31, %al
	jne	.L1818
	cmpq	$1, 80(%rsp)
	jne	.L1819
	movq	%rdi, 16(%rsp)
	movq	%rdx, 8(%rsp)
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	movq	%rsp, %rsi
	leaq	_Z24axpb_simd_parallel_m_mnmILi32EEl14axpb_options_tfPff._omp_fn.0(%rip), %rdi
	movq	%rax, (%rsp)
	vmovss	%xmm1, 28(%rsp)
	vmovss	%xmm0, 24(%rsp)
	call	GOMP_parallel@PLT
	movq	40(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L1820
	xorl	%eax, %eax
	addq	$56, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L1818:
	.cfi_restore_state
	leaq	.LC156(%rip), %rcx
	movl	$288, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC155(%rip), %rdi
	call	__assert_fail@PLT
.L1820:
	call	__stack_chk_fail@PLT
.L1819:
	leaq	.LC156(%rip), %rcx
	movl	$289, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5811:
	.size	_Z24axpb_simd_parallel_m_mnmILi32EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi32EEl14axpb_options_tfPff
	.section	.rodata._Z24axpb_simd_parallel_m_mnmILi48EEl14axpb_options_tfPff.str1.8,"aMS",@progbits,1
	.align 8
.LC157:
	.string	"long int axpb_simd_parallel_m_mnm(axpb_options_t, float, float*, float) [with int c = 48]"
	.section	.text._Z24axpb_simd_parallel_m_mnmILi48EEl14axpb_options_tfPff,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi48EEl14axpb_options_tfPff,comdat
	.p2align 4
	.weak	_Z24axpb_simd_parallel_m_mnmILi48EEl14axpb_options_tfPff
	.type	_Z24axpb_simd_parallel_m_mnmILi48EEl14axpb_options_tfPff, @function
_Z24axpb_simd_parallel_m_mnmILi48EEl14axpb_options_tfPff:
.LFB5859:
	.cfi_startproc
	endbr64
	subq	$56, %rsp
	.cfi_def_cfa_offset 64
	movq	104(%rsp), %rdx
	movq	%fs:40, %rax
	movq	%rax, 40(%rsp)
	xorl	%eax, %eax
	movabsq	$-6148914691236517205, %rax
	imulq	%rdx, %rax
	movabsq	$3074457345618258592, %rcx
	movq	112(%rsp), %rsi
	addq	%rcx, %rax
	movabsq	$384307168202282324, %rcx
	rorx	$4, %rax, %rax
	cmpq	%rcx, %rax
	ja	.L1826
	cmpq	$1, 80(%rsp)
	jne	.L1827
	movq	%rdi, 16(%rsp)
	movq	%rsi, 8(%rsp)
	movq	%rdx, (%rsp)
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	movq	%rsp, %rsi
	leaq	_Z24axpb_simd_parallel_m_mnmILi48EEl14axpb_options_tfPff._omp_fn.0(%rip), %rdi
	vmovss	%xmm1, 28(%rsp)
	vmovss	%xmm0, 24(%rsp)
	call	GOMP_parallel@PLT
	movq	40(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L1828
	xorl	%eax, %eax
	addq	$56, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L1826:
	.cfi_restore_state
	leaq	.LC157(%rip), %rcx
	movl	$288, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC155(%rip), %rdi
	call	__assert_fail@PLT
.L1828:
	call	__stack_chk_fail@PLT
.L1827:
	leaq	.LC157(%rip), %rcx
	movl	$289, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5859:
	.size	_Z24axpb_simd_parallel_m_mnmILi48EEl14axpb_options_tfPff, .-_Z24axpb_simd_parallel_m_mnmILi48EEl14axpb_options_tfPff
	.section	.text._Z24axpb_simd_parallel_m_mnmILi32EEl14axpb_options_tfPff._omp_fn.0,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi32EEl14axpb_options_tfPff,comdat
	.p2align 4
	.type	_Z24axpb_simd_parallel_m_mnmILi32EEl14axpb_options_tfPff._omp_fn.0, @function
_Z24axpb_simd_parallel_m_mnmILi32EEl14axpb_options_tfPff._omp_fn.0:
.LFB5896:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-64, %rsp
	subq	$2176, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	%rdi, (%rsp)
	movq	%fs:40, %rax
	movq	%rax, 2168(%rsp)
	xorl	%eax, %eax
	call	omp_get_num_threads@PLT
	movslq	%eax, %rbx
	call	omp_get_thread_num@PLT
	movq	(%rsp), %rdi
	movslq	%eax, %rsi
	movq	(%rdi), %rax
	testq	%rax, %rax
	leaq	15(%rax), %rdx
	cmovns	%rax, %rdx
	sarq	$4, %rdx
	leaq	62(%rdx), %rax
	addq	$31, %rdx
	cmovns	%rdx, %rax
	sarq	$5, %rax
	cqto
	idivq	%rbx
	cmpq	%rdx, %rsi
	jl	.L1830
.L1840:
	imulq	%rax, %rsi
	addq	%rsi, %rdx
	addq	%rdx, %rax
	cmpq	%rax, %rdx
	jge	.L1829
	movq	%rdx, %r12
	vmovss	28(%rdi), %xmm4
	salq	$11, %rdx
	salq	$5, %rax
	addq	16(%rdi), %rdx
	movq	8(%rdi), %r14
	vbroadcastss	24(%rdi), %zmm2
	vmovss	%xmm4, 108(%rsp)
	salq	$5, %r12
	movq	%rax, %rbx
	movq	%rdx, %r15
	leaq	112(%rsp), %r13
.L1835:
	movl	$2048, %edx
	movq	%r15, %rsi
	movq	%r13, %rdi
	vmovaps	%zmm2, (%rsp)
	vzeroupper
	call	memcpy@PLT
#APP
# 297 "axpb.cc" 1
	# axpb_simd_parallel_m_mnm<$32>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%r14, %r14
	vmovaps	(%rsp), %zmm2
	jle	.L1832
	vbroadcastss	108(%rsp), %zmm1
	leaq	-1(%r14), %rdi
	movl	$1, %esi
	leaq	2160(%rsp), %rdx
	xorl	%ecx, %ecx
	cmpq	$2, %r14
	jle	.L1837
.L1833:
	movq	%r13, %rax
	.p2align 4,,10
	.p2align 3
.L1839:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rax, %rdx
	jne	.L1839
	leaq	1(%rsi), %rcx
	addq	$2, %rsi
	cmpq	%rsi, %rdi
	jg	.L1833
	.p2align 4,,10
	.p2align 3
.L1837:
	movq	%r13, %rax
	.p2align 4,,10
	.p2align 3
.L1836:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rax, %rdx
	jne	.L1836
	incq	%rcx
	cmpq	%rcx, %r14
	jg	.L1837
.L1832:
	vmovaps	%zmm2, (%rsp)
#APP
# 303 "axpb.cc" 1
	# axpb_simd_parallel_m_mnm<$32>: ax+c inner loop end
# 0 "" 2
#NO_APP
	movl	$2048, %edx
	movq	%r13, %rsi
	movq	%r15, %rdi
	vzeroupper
	addq	$32, %r12
	call	memcpy@PLT
	addq	$2048, %r15
	cmpq	%r12, %rbx
	vmovaps	(%rsp), %zmm2
	jg	.L1835
	vzeroupper
.L1829:
	movq	2168(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L1849
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1830:
	.cfi_restore_state
	incq	%rax
	xorl	%edx, %edx
	jmp	.L1840
.L1849:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE5896:
	.size	_Z24axpb_simd_parallel_m_mnmILi32EEl14axpb_options_tfPff._omp_fn.0, .-_Z24axpb_simd_parallel_m_mnmILi32EEl14axpb_options_tfPff._omp_fn.0
	.section	.text._Z24axpb_simd_parallel_m_mnmILi48EEl14axpb_options_tfPff._omp_fn.0,"axG",@progbits,_Z24axpb_simd_parallel_m_mnmILi48EEl14axpb_options_tfPff,comdat
	.p2align 4
	.type	_Z24axpb_simd_parallel_m_mnmILi48EEl14axpb_options_tfPff._omp_fn.0, @function
_Z24axpb_simd_parallel_m_mnmILi48EEl14axpb_options_tfPff._omp_fn.0:
.LFB5912:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	.cfi_offset 15, -24
	movq	%rdi, %r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-64, %rsp
	subq	$3200, %rsp
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	%fs:40, %rax
	movq	%rax, 3192(%rsp)
	xorl	%eax, %eax
	call	omp_get_num_threads@PLT
	movslq	%eax, %rbx
	call	omp_get_thread_num@PLT
	movslq	%eax, %rcx
	movq	(%r15), %rax
	movabsq	$3074457345618258603, %rdx
	testq	%rax, %rax
	leaq	15(%rax), %rsi
	cmovns	%rax, %rsi
	sarq	$4, %rsi
	addq	$47, %rsi
	movq	%rsi, %rax
	imulq	%rdx
	sarq	$63, %rsi
	movq	%rdx, %rax
	sarq	$3, %rax
	subq	%rsi, %rax
	cqto
	idivq	%rbx
	cmpq	%rdx, %rcx
	jl	.L1851
.L1861:
	imulq	%rax, %rcx
	addq	%rcx, %rdx
	addq	%rdx, %rax
	cmpq	%rax, %rdx
	jge	.L1850
	leaq	(%rdx,%rdx,2), %rdx
	movq	%rdx, %rbx
	vmovss	28(%r15), %xmm4
	leaq	(%rax,%rax,2), %r12
	salq	$10, %rdx
	salq	$4, %r12
	addq	16(%r15), %rdx
	movq	8(%r15), %r14
	vbroadcastss	24(%r15), %zmm2
	movq	%r12, 96(%rsp)
	vmovss	%xmm4, 108(%rsp)
	salq	$4, %rbx
	movq	%rdx, %r12
	leaq	112(%rsp), %r13
.L1856:
	movl	$3072, %edx
	movq	%r12, %rsi
	movq	%r13, %rdi
	vmovaps	%zmm2, (%rsp)
	vzeroupper
	call	memcpy@PLT
#APP
# 297 "axpb.cc" 1
	# axpb_simd_parallel_m_mnm<$48>: ax+c inner loop begin
# 0 "" 2
#NO_APP
	testq	%r14, %r14
	vmovaps	(%rsp), %zmm2
	jle	.L1853
	vbroadcastss	108(%rsp), %zmm1
	leaq	-1(%r14), %rdi
	movl	$1, %esi
	leaq	3184(%rsp), %rdx
	xorl	%ecx, %ecx
	cmpq	$2, %r14
	jle	.L1858
.L1854:
	movq	%r13, %rax
	.p2align 4,,10
	.p2align 3
.L1860:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vfmadd132ps	%zmm2, %zmm1, %zmm0
	vmovups	%zmm0, -64(%rax)
	cmpq	%rax, %rdx
	jne	.L1860
	leaq	1(%rsi), %rcx
	addq	$2, %rsi
	cmpq	%rsi, %rdi
	jg	.L1854
	.p2align 4,,10
	.p2align 3
.L1858:
	movq	%r13, %rax
	.p2align 4,,10
	.p2align 3
.L1857:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rax, %rdx
	jne	.L1857
	incq	%rcx
	cmpq	%rcx, %r14
	jg	.L1858
.L1853:
	vmovaps	%zmm2, (%rsp)
#APP
# 303 "axpb.cc" 1
	# axpb_simd_parallel_m_mnm<$48>: ax+c inner loop end
# 0 "" 2
#NO_APP
	movl	$3072, %edx
	movq	%r13, %rsi
	movq	%r12, %rdi
	vzeroupper
	addq	$48, %rbx
	call	memcpy@PLT
	addq	$3072, %r12
	cmpq	%rbx, 96(%rsp)
	vmovaps	(%rsp), %zmm2
	jg	.L1856
	vzeroupper
.L1850:
	movq	3192(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L1870
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1851:
	.cfi_restore_state
	incq	%rax
	xorl	%edx, %edx
	jmp	.L1861
.L1870:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE5912:
	.size	_Z24axpb_simd_parallel_m_mnmILi48EEl14axpb_options_tfPff._omp_fn.0, .-_Z24axpb_simd_parallel_m_mnmILi48EEl14axpb_options_tfPff._omp_fn.0
	.section	.rodata.str1.8
	.align 8
.LC158:
	.string	"long int axpb_scalar(axpb_options_t, float, float*, float)"
	.section	.rodata.str1.1
.LC159:
	.string	"opt.m == 1"
	.text
	.p2align 4
	.globl	_Z11axpb_scalar14axpb_options_tfPff
	.type	_Z11axpb_scalar14axpb_options_tfPff, @function
_Z11axpb_scalar14axpb_options_tfPff:
.LFB5652:
	.cfi_startproc
	endbr64
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	cmpq	$1, 56(%rsp)
	jne	.L1878
	cmpq	$1, 32(%rsp)
	jne	.L1879
	movq	64(%rsp), %rdx
	vmovss	(%rdi), %xmm2
#APP
# 92 "axpb.cc" 1
	# axpb_scalar: ax+b loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L1874
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L1875:
	incq	%rax
	vfmadd132ss	%xmm0, %xmm1, %xmm2
	cmpq	%rax, %rdx
	jne	.L1875
.L1874:
#APP
# 96 "axpb.cc" 1
	# axpb_scalar: ax+b loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	vmovss	%xmm2, (%rdi)
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L1878:
	.cfi_restore_state
	leaq	.LC158(%rip), %rcx
	movl	$88, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC159(%rip), %rdi
	call	__assert_fail@PLT
.L1879:
	leaq	.LC158(%rip), %rcx
	movl	$89, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5652:
	.size	_Z11axpb_scalar14axpb_options_tfPff, .-_Z11axpb_scalar14axpb_options_tfPff
	.section	.rodata.str1.8
	.align 8
.LC160:
	.string	"long int axpb_simd(axpb_options_t, float, float*, float)"
	.section	.rodata.str1.1
.LC161:
	.string	"opt.m == L"
	.text
	.p2align 4
	.globl	_Z9axpb_simd14axpb_options_tfPff
	.type	_Z9axpb_simd14axpb_options_tfPff, @function
_Z9axpb_simd14axpb_options_tfPff:
.LFB5653:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-64, %rsp
	cmpq	$16, 56(%rbp)
	jne	.L1887
	cmpq	$1, 32(%rbp)
	jne	.L1888
	movq	64(%rbp), %rdx
	vmovups	(%rdi), %zmm2
#APP
# 118 "axpb.cc" 1
	# axpb_simd: ax+b loop begin
# 0 "" 2
#NO_APP
	testq	%rdx, %rdx
	jle	.L1883
	vbroadcastss	%xmm0, %zmm0
	vbroadcastss	%xmm1, %zmm1
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L1884:
	incq	%rax
	vfmadd132ps	%zmm0, %zmm1, %zmm2
	cmpq	%rax, %rdx
	jne	.L1884
.L1883:
#APP
# 122 "axpb.cc" 1
	# axpb_simd: ax+b loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	vmovups	%zmm2, (%rdi)
	vzeroupper
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1887:
	.cfi_restore_state
	leaq	.LC160(%rip), %rcx
	movl	$113, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC161(%rip), %rdi
	call	__assert_fail@PLT
.L1888:
	leaq	.LC160(%rip), %rcx
	movl	$114, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5653:
	.size	_Z9axpb_simd14axpb_options_tfPff, .-_Z9axpb_simd14axpb_options_tfPff
	.section	.rodata.str1.8
	.align 8
.LC162:
	.string	"long int axpb_simd_m(axpb_options_t, float, float*, float)"
	.text
	.p2align 4
	.globl	_Z11axpb_simd_m14axpb_options_tfPff
	.type	_Z11axpb_simd_m14axpb_options_tfPff, @function
_Z11axpb_simd_m14axpb_options_tfPff:
.LFB5655:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	56(%rbp), %rax
	andq	$-64, %rsp
	testb	$15, %al
	jne	.L1900
	cmpq	$1, 32(%rbp)
	jne	.L1901
	movq	64(%rbp), %r8
#APP
# 194 "axpb.cc" 1
	# axpb_simd_m: ax+c loop begin
# 0 "" 2
#NO_APP
	testq	%r8, %r8
	jle	.L1895
	testq	%rax, %rax
	leaq	15(%rax), %rcx
	cmovns	%rax, %rcx
	sarq	$4, %rcx
	cmpq	$15, %rax
	jle	.L1895
	xorl	%esi, %esi
	vbroadcastss	%xmm0, %zmm2
	vbroadcastss	%xmm1, %zmm1
	.p2align 4,,10
	.p2align 3
.L1896:
	movq	%rdi, %rax
	xorl	%edx, %edx
	.p2align 4,,10
	.p2align 3
.L1894:
	vmovaps	%zmm2, %zmm0
	vfmadd132ps	(%rax), %zmm1, %zmm0
	incq	%rdx
	addq	$64, %rax
	vmovups	%zmm0, -64(%rax)
	cmpq	%rcx, %rdx
	jl	.L1894
	incq	%rsi
	cmpq	%rsi, %r8
	jne	.L1896
	vzeroupper
.L1895:
#APP
# 200 "axpb.cc" 1
	# axpb_simd_m: ax+c loop end
# 0 "" 2
#NO_APP
	xorl	%eax, %eax
	leave
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L1900:
	.cfi_restore_state
	leaq	.LC162(%rip), %rcx
	movl	$190, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC2(%rip), %rdi
	call	__assert_fail@PLT
.L1901:
	leaq	.LC162(%rip), %rcx
	movl	$191, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC4(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5655:
	.size	_Z11axpb_simd_m14axpb_options_tfPff, .-_Z11axpb_simd_m14axpb_options_tfPff
	.section	.rodata.str1.1
.LC163:
	.string	"clock_gettime"
.LC164:
	.string	"clock.h"
	.section	.rodata.str1.8
	.align 8
.LC165:
	.string	"%s:%d:cpu_clock_counter_get: the caller thread (%ld) is invalid (!= %ld)\n"
	.section	.rodata.str1.1
.LC166:
	.string	"read"
	.section	.rodata.str1.8
	.align 8
.LC167:
	.string	"long long int cpu_clock_counter_get(cpu_clock_counter_t)"
	.section	.rodata.str1.1
.LC168:
	.string	"rd == sizeof(long long)"
	.text
	.p2align 4
	.type	_ZL18clock_counters_get16clock_counters_t.isra.0, @function
_ZL18clock_counters_get16clock_counters_t.isra.0:
.LFB5918:
	.cfi_startproc
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	movq	%rdi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	movl	%edx, %ebp
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	movq	%rsi, %rbx
	subq	$48, %rsp
	.cfi_def_cfa_offset 80
	movq	%fs:40, %rax
	movq	%rax, 40(%rsp)
	xorl	%eax, %eax
#APP
# 31 "clock.h" 1
	rdtsc;shlq $32,%rdx;orq %rdx,%rax
# 0 "" 2
#NO_APP
	movq	%rax, (%rdi)
	leaq	16(%rsp), %rsi
	xorl	%edi, %edi
	call	clock_gettime@PLT
	cmpl	$-1, %eax
	je	.L1914
	imulq	$1000000000, 16(%rsp), %rax
	addq	24(%rsp), %rax
	movq	%rax, 16(%r12)
	call	pthread_self@PLT
	cmpq	%rax, %rbx
	jne	.L1915
	cmpl	$-1, %ebp
	je	.L1916
	leaq	8(%rsp), %rsi
	movl	$8, %edx
	movl	%ebp, %edi
	call	read@PLT
	cmpq	$-1, %rax
	je	.L1917
	cmpq	$8, %rax
	jne	.L1918
.L1907:
	movq	8(%rsp), %rax
.L1905:
	movq	%rax, 8(%r12)
	movq	40(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L1919
	addq	$48, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	movq	%r12, %rax
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
.L1916:
	.cfi_restore_state
	movq	$0, 8(%rsp)
	jmp	.L1907
.L1915:
	pushq	%rdx
	.cfi_def_cfa_offset 88
	movq	stderr(%rip), %rdi
	movq	%rax, %r9
	pushq	%rbx
	.cfi_def_cfa_offset 96
	leaq	.LC164(%rip), %rcx
	movl	$1, %esi
	movl	$132, %r8d
	leaq	.LC165(%rip), %rdx
	xorl	%eax, %eax
	call	__fprintf_chk@PLT
	popq	%rcx
	.cfi_def_cfa_offset 88
	popq	%rsi
	.cfi_def_cfa_offset 80
	orq	$-1, %rax
	jmp	.L1905
.L1914:
	leaq	.LC163(%rip), %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L1919:
	call	__stack_chk_fail@PLT
.L1918:
	leaq	.LC167(%rip), %rcx
	movl	$145, %edx
	leaq	.LC164(%rip), %rsi
	leaq	.LC168(%rip), %rdi
	call	__assert_fail@PLT
.L1917:
	leaq	.LC166(%rip), %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
	.cfi_endproc
.LFE5918:
	.size	_ZL18clock_counters_get16clock_counters_t.isra.0, .-_ZL18clock_counters_get16clock_counters_t.isra.0
	.section	.rodata.str1.8
	.align 8
.LC169:
	.string	"long int axpb(axpb_options_t, float, float*, float)"
	.section	.rodata.str1.1
.LC170:
	.string	"c > 0"
.LC171:
	.string	"c < table_sz"
	.text
	.p2align 4
	.globl	_Z4axpb14axpb_options_tfPff
	.type	_Z4axpb14axpb_options_tfPff, @function
_Z4axpb14axpb_options_tfPff:
.LFB5660:
	.cfi_startproc
	endbr64
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	movq	48(%rsp), %rax
	testq	%rax, %rax
	jle	.L1924
	cmpq	$49, %rax
	jg	.L1925
	movslq	24(%rsp), %rdx
	leaq	(%rax,%rax,8), %rax
	addq	%rdx, %rax
	leaq	axpb_funs_table(%rip), %rdx
	movq	(%rdx,%rax,8), %rax
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	jmp	*%rax
.L1924:
	.cfi_restore_state
	leaq	.LC169(%rip), %rcx
	movl	$486, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC170(%rip), %rdi
	call	__assert_fail@PLT
.L1925:
	leaq	.LC169(%rip), %rcx
	movl	$487, %edx
	leaq	.LC1(%rip), %rsi
	leaq	.LC171(%rip), %rdi
	call	__assert_fail@PLT
	.cfi_endproc
.LFE5660:
	.size	_Z4axpb14axpb_options_tfPff, .-_Z4axpb14axpb_options_tfPff
	.section	.rodata.str1.8
	.align 8
.LC172:
	.string	"%s:%d:parse_algo: invalid algo %s\n"
	.text
	.p2align 4
	.globl	_Z10parse_algoPKc
	.type	_Z10parse_algoPKc, @function
_Z10parse_algoPKc:
.LFB5661:
	.cfi_startproc
	endbr64
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	movq	%rdi, %rsi
	movq	%rdi, %rbx
	movq	8+_ZL10algo_table(%rip), %rdi
	call	strcmp@PLT
	testl	%eax, %eax
	je	.L1930
	movq	24+_ZL10algo_table(%rip), %rdi
	movq	%rbx, %rsi
	call	strcmp@PLT
	testl	%eax, %eax
	je	.L1931
	movq	40+_ZL10algo_table(%rip), %rdi
	movq	%rbx, %rsi
	call	strcmp@PLT
	testl	%eax, %eax
	je	.L1932
	movq	56+_ZL10algo_table(%rip), %rdi
	movq	%rbx, %rsi
	call	strcmp@PLT
	testl	%eax, %eax
	je	.L1933
	movq	72+_ZL10algo_table(%rip), %rdi
	movq	%rbx, %rsi
	call	strcmp@PLT
	testl	%eax, %eax
	je	.L1934
	movq	88+_ZL10algo_table(%rip), %rdi
	movq	%rbx, %rsi
	call	strcmp@PLT
	testl	%eax, %eax
	je	.L1935
	movq	104+_ZL10algo_table(%rip), %rdi
	movq	%rbx, %rsi
	call	strcmp@PLT
	testl	%eax, %eax
	je	.L1936
	movq	120+_ZL10algo_table(%rip), %rdi
	movq	%rbx, %rsi
	call	strcmp@PLT
	testl	%eax, %eax
	je	.L1937
	movq	136+_ZL10algo_table(%rip), %rdi
	movq	%rbx, %rsi
	call	strcmp@PLT
	testl	%eax, %eax
	je	.L1940
	movq	stderr(%rip), %rdi
	movq	%rbx, %r9
	movl	$524, %r8d
	leaq	.LC1(%rip), %rcx
	leaq	.LC172(%rip), %rdx
	movl	$1, %esi
	xorl	%eax, %eax
	call	__fprintf_chk@PLT
	movl	$9, %eax
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L1930:
	.cfi_restore_state
	leaq	_ZL10algo_table(%rip), %rax
	movl	(%rax), %eax
.L1941:
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L1931:
	.cfi_restore_state
	leaq	16+_ZL10algo_table(%rip), %rax
	movl	(%rax), %eax
	jmp	.L1941
.L1932:
	leaq	32+_ZL10algo_table(%rip), %rax
	movl	(%rax), %eax
	jmp	.L1941
.L1933:
	leaq	48+_ZL10algo_table(%rip), %rax
	movl	(%rax), %eax
	jmp	.L1941
.L1934:
	leaq	64+_ZL10algo_table(%rip), %rax
	movl	(%rax), %eax
	jmp	.L1941
.L1935:
	leaq	80+_ZL10algo_table(%rip), %rax
	movl	(%rax), %eax
	jmp	.L1941
.L1936:
	leaq	96+_ZL10algo_table(%rip), %rax
	movl	(%rax), %eax
	jmp	.L1941
.L1937:
	leaq	112+_ZL10algo_table(%rip), %rax
	movl	(%rax), %eax
	jmp	.L1941
.L1940:
	leaq	128+_ZL10algo_table(%rip), %rax
	movl	(%rax), %eax
	jmp	.L1941
	.cfi_endproc
.LFE5661:
	.size	_Z10parse_algoPKc, .-_Z10parse_algoPKc
	.section	.rodata.str1.1
.LC173:
	.string	"a:b:w:c:m:n:s:h"
	.section	.rodata.str1.8
	.align 8
.LC174:
	.string	"bug:%s:%d: should handle option %s\n"
	.section	.rodata.str1.1
.LC175:
	.string	" algo = %s\n"
	.section	.rodata.str1.8
	.align 8
.LC176:
	.string	"    bs = %ld (cuda block size)\n"
	.align 8
.LC177:
	.string	"    c = %ld (the number of variables to update in the inner loop)\n"
	.align 8
.LC178:
	.string	"    m = %ld (the number of FP numbers to update)\n"
	.align 8
.LC179:
	.string	"    n = %ld (the number of times to update each variable)\n"
	.align 8
.LC180:
	.string	"    L = %d (SIMD lanes on the CPU)\n"
	.section	.rodata.str1.1
.LC181:
	.string	"malloc"
.LC182:
	.string	"perf_event_open"
.LC183:
	.string	"ioctl"
	.section	.rodata.str1.8
	.align 8
.LC184:
	.string	"%s:%d:warning: the environment does not support perf_event. CPU clock cannot be obtained\n"
	.section	.rodata.str1.1
.LC185:
	.string	"CLOCK_ADJUST_CPU"
.LC186:
	.string	"CLOCK_ADJUST_REF"
	.section	.rodata.str1.8
	.align 8
.LC187:
	.string	"get cpu cycles by ref cycles x %f / %f\n"
	.section	.rodata.str1.1
.LC188:
	.string	"%ld nsec\n"
.LC189:
	.string	"%ld ref clocks\n"
.LC190:
	.string	"%ld cpu clocks\n"
	.section	.rodata.str1.8
	.align 8
.LC191:
	.string	"%f nsec       for performing x=ax+b for %ld variables once\n"
	.align 8
.LC192:
	.string	"%f ref clocks for performing x=ax+b for %ld variables once\n"
	.align 8
.LC193:
	.string	"%f cpu clocks for performing x=ax+b for %ld variables once\n"
	.section	.rodata.str1.1
.LC194:
	.string	"%f flops/nsec\n"
.LC195:
	.string	"%f flops/ref clock\n"
.LC196:
	.string	"%f flops/cpu clock\n"
.LC197:
	.string	"-------- cpu clocks"
	.section	.rodata.str1.8
	.align 8
.LC198:
	.string	"-------- cpu clocks for performing x=ax+b for %ld variables once\n"
	.section	.rodata.str1.1
.LC199:
	.string	"-------- flops/cpu clock"
.LC200:
	.string	"X[%ld] = %f\n"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB5666:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	leaq	.L1947(%rip), %r15
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	movq	%rsi, %r14
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	xorl	%r13d, %r13d
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	movl	$1, %r12d
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movl	$1, %ebp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movl	$1, %ebx
	subq	$344, %rsp
	.cfi_def_cfa_offset 400
	movl	%edi, 8(%rsp)
	movq	%fs:40, %rax
	movq	%rax, 328(%rsp)
	xorl	%eax, %eax
	movq	(%rsi), %rax
	movq	$1000000, 24(%rsp)
	movq	%rax, 48(%rsp)
	movabsq	$76843802738543, %rax
	movq	%rax, 40(%rsp)
	leaq	.LC152(%rip), %rax
	movq	%rax, 32(%rsp)
	leaq	208(%rsp), %rax
	movq	$32, 56(%rsp)
	movq	%rax, 16(%rsp)
.L1956:
	movq	16(%rsp), %r8
	movl	8(%rsp), %edi
	leaq	_ZL12long_options(%rip), %rcx
	leaq	.LC173(%rip), %rdx
	movq	%r14, %rsi
	movl	$0, 208(%rsp)
	call	getopt_long@PLT
	cmpl	$-1, %eax
	je	.L1943
	testl	%eax, %eax
	je	.L1944
	subl	$97, %eax
	cmpl	$22, %eax
	ja	.L1945
	movslq	(%r15,%rax,4), %rax
	addq	%r15, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L1947:
	.long	.L1954-.L1947
	.long	.L1953-.L1947
	.long	.L1952-.L1947
	.long	.L1945-.L1947
	.long	.L1945-.L1947
	.long	.L1945-.L1947
	.long	.L1945-.L1947
	.long	.L1990-.L1947
	.long	.L1945-.L1947
	.long	.L1945-.L1947
	.long	.L1945-.L1947
	.long	.L1945-.L1947
	.long	.L1950-.L1947
	.long	.L1949-.L1947
	.long	.L1945-.L1947
	.long	.L1945-.L1947
	.long	.L1945-.L1947
	.long	.L1945-.L1947
	.long	.L1948-.L1947
	.long	.L1945-.L1947
	.long	.L1945-.L1947
	.long	.L1945-.L1947
	.long	.L1946-.L1947
	.section	.text.startup
.L1990:
	movl	$1, %r13d
	jmp	.L1956
.L1946:
	movq	optarg(%rip), %rdi
	movl	$10, %edx
	xorl	%esi, %esi
	call	strtol@PLT
	movq	%rax, 56(%rsp)
	jmp	.L1956
.L1948:
	movq	optarg(%rip), %rdi
	movl	$10, %edx
	xorl	%esi, %esi
	call	strtol@PLT
	movq	%rax, 40(%rsp)
	jmp	.L1956
.L1949:
	movq	optarg(%rip), %rdi
	movl	$10, %edx
	xorl	%esi, %esi
	call	strtol@PLT
	movq	%rax, 24(%rsp)
	jmp	.L1956
.L1950:
	movq	optarg(%rip), %rdi
	movl	$10, %edx
	xorl	%esi, %esi
	call	strtol@PLT
	movq	%rax, %rbx
	jmp	.L1956
.L1952:
	movq	optarg(%rip), %rdi
	movl	$10, %edx
	xorl	%esi, %esi
	call	strtol@PLT
	movq	%rax, %r12
	jmp	.L1956
.L1953:
	movq	optarg(%rip), %rdi
	movl	$10, %edx
	xorl	%esi, %esi
	call	strtol@PLT
	movq	%rax, %rbp
	jmp	.L1956
.L1954:
	movq	optarg(%rip), %rdi
	call	strdup@PLT
	movq	%rax, 32(%rsp)
	jmp	.L1956
.L1945:
	movq	48(%rsp), %rdi
	call	_ZL5usagePKc
.L1955:
	movq	$1, 8(%rsp)
	movl	$1, %r8d
	movl	$9, %r15d
.L1957:
	movl	%r13d, 200(%rsp)
	movl	%r8d, 204(%rsp)
	movq	200(%rsp), %rcx
	testq	%rcx, %rcx
	jne	.L2012
	movq	32(%rsp), %rdx
	leaq	.LC175(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	movq	%rcx, 48(%rsp)
	call	__printf_chk@PLT
	movq	%rbp, %rdx
	leaq	.LC176(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movq	%r12, %rdx
	leaq	.LC177(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movq	%rbx, %rdx
	leaq	.LC178(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movq	24(%rsp), %rdx
	leaq	.LC179(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movl	$16, %edx
	leaq	.LC180(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movq	stdout(%rip), %rdi
	leaq	322(%rsp), %r14
	call	fflush@PLT
	movq	40(%rsp), %rdi
	movq	%rdi, %rax
	sarq	$16, %rax
	movw	%ax, 322(%rsp)
	movq	%rdi, %rax
	sarq	$8, %rax
	movw	%di, 326(%rsp)
	movq	%r14, %rdi
	movw	%ax, 324(%rsp)
	call	erand48@PLT
	movq	%r14, %rdi
	vcvtsd2ss	%xmm0, %xmm0, %xmm5
	vmovss	%xmm5, 68(%rsp)
	call	erand48@PLT
	leaq	0(,%rbx,4), %rsi
	movl	$4, %edi
	vcvtsd2ss	%xmm0, %xmm0, %xmm6
	vmovss	%xmm6, 88(%rsp)
	call	aligned_alloc@PLT
	testq	%rax, %rax
	movq	48(%rsp), %rcx
	movq	%rax, %r13
	je	.L2013
	testq	%rbx, %rbx
	jle	.L1972
	.p2align 4,,10
	.p2align 3
.L1973:
	vxorps	%xmm3, %xmm3, %xmm3
	vcvtsi2ssq	%rcx, %xmm3, %xmm0
	vmovss	%xmm0, 0(%r13,%rcx,4)
	incq	%rcx
	cmpq	%rbx, %rcx
	jne	.L1973
.L1972:
	movq	24(%rsp), %rax
	imulq	%rbx, %rax
	addq	%rax, %rax
	movq	%rax, 72(%rsp)
	call	pthread_self@PLT
	movq	%rax, 48(%rsp)
	leaq	224(%rsp), %rsi
	xorl	%ecx, %ecx
	xorl	%edx, %edx
.L1974:
	movl	%edx, %eax
	addl	$32, %edx
	movq	%rcx, (%rsi,%rax)
	movq	%rcx, 8(%rsi,%rax)
	movq	%rcx, 16(%rsi,%rax)
	movq	%rcx, 24(%rsi,%rax)
	cmpl	$96, %edx
	jb	.L1974
	movabsq	$481036337152, %rax
	movq	16(%rsp), %rsi
	movq	%rax, 208(%rsp)
	xorl	%r9d, %r9d
	xorl	%edx, %edx
	xorl	%eax, %eax
	movl	$-1, %r8d
	movl	$-1, %ecx
	movl	$298, %edi
	orb	$97, 248(%rsp)
	movq	$0, 216(%rsp)
	call	syscall@PLT
	cmpl	$-1, %eax
	je	.L2014
	movl	%eax, 92(%rsp)
	movq	%rax, 80(%rsp)
	xorl	%edx, %edx
	movl	%eax, %edi
	movl	$9219, %esi
	xorl	%eax, %eax
	call	ioctl@PLT
	incl	%eax
	movq	80(%rsp), %rcx
	movl	92(%rsp), %r8d
	je	.L1985
	xorl	%edx, %edx
	movl	%ecx, %edi
	xorl	%eax, %eax
	movl	$9216, %esi
	movl	%r8d, 92(%rsp)
	movq	%rcx, 80(%rsp)
	call	ioctl@PLT
	incl	%eax
	movq	80(%rsp), %rcx
	movl	92(%rsp), %r8d
	je	.L1985
.L1986:
	movq	48(%rsp), %rsi
	leaq	96(%rsp), %rdi
	movl	%r8d, %edx
	movl	%r8d, 80(%rsp)
	call	_ZL18clock_counters_get16clock_counters_t.isra.0
	movq	32(%rsp), %rax
	movl	%r15d, 136(%rsp)
	movq	%rax, 128(%rsp)
	movq	56(%rsp), %rax
	movq	%rbp, 144(%rsp)
	movq	%rax, 152(%rsp)
	movq	24(%rsp), %rax
	movq	%r12, 160(%rsp)
	movq	%rax, 176(%rsp)
	movq	40(%rsp), %rax
	movq	%rbx, 168(%rsp)
	movq	%rax, 184(%rsp)
	movq	8(%rsp), %rax
	movq	%r13, %rdi
	movq	%rax, 192(%rsp)
	pushq	200(%rsp)
	.cfi_def_cfa_offset 408
	pushq	200(%rsp)
	.cfi_def_cfa_offset 416
	pushq	200(%rsp)
	.cfi_def_cfa_offset 424
	pushq	200(%rsp)
	.cfi_def_cfa_offset 432
	pushq	200(%rsp)
	.cfi_def_cfa_offset 440
	pushq	200(%rsp)
	.cfi_def_cfa_offset 448
	pushq	200(%rsp)
	.cfi_def_cfa_offset 456
	pushq	200(%rsp)
	.cfi_def_cfa_offset 464
	pushq	200(%rsp)
	.cfi_def_cfa_offset 472
	pushq	200(%rsp)
	.cfi_def_cfa_offset 480
	vmovss	168(%rsp), %xmm1
	vmovss	148(%rsp), %xmm0
	call	_Z4axpb14axpb_options_tfPff
	movl	160(%rsp), %r8d
	movq	128(%rsp), %rsi
	movq	96(%rsp), %rdi
	addq	$80, %rsp
	.cfi_def_cfa_offset 400
	movl	%r8d, %edx
	call	_ZL18clock_counters_get16clock_counters_t.isra.0
	movq	216(%rsp), %rax
	movq	208(%rsp), %r12
	movq	224(%rsp), %rbp
	subq	96(%rsp), %r12
	subq	112(%rsp), %rbp
	subq	104(%rsp), %rax
	movq	%rax, %r15
	je	.L2015
	movq	%rbp, %rdx
	leaq	.LC188(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movq	%r12, %rdx
	leaq	.LC189(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	vxorpd	%xmm7, %xmm7, %xmm7
	vcvtsi2sdq	%r12, %xmm7, %xmm0
	vmovsd	%xmm0, 16(%rsp)
.L1984:
	movq	%r15, %rdx
	leaq	.LC190(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movl	$10, %edi
	call	putchar@PLT
	vxorpd	%xmm4, %xmm4, %xmm4
	vcvtsi2sdq	%rbp, %xmm4, %xmm2
	vcvtsi2sdq	24(%rsp), %xmm4, %xmm0
	movq	%rbx, %rdx
	leaq	.LC191(%rip), %rsi
	movl	$1, %edi
	vmovsd	%xmm0, 24(%rsp)
	vdivsd	%xmm0, %xmm2, %xmm0
	movl	$1, %eax
	vmovsd	%xmm2, 40(%rsp)
	call	__printf_chk@PLT
	vmovsd	16(%rsp), %xmm5
	movq	%rbx, %rdx
	leaq	.LC192(%rip), %rsi
	movl	$1, %edi
	movl	$1, %eax
	vdivsd	24(%rsp), %xmm5, %xmm0
	call	__printf_chk@PLT
	vxorpd	%xmm4, %xmm4, %xmm4
	vcvtsi2sdq	%r15, %xmm4, %xmm1
	movq	%rbx, %rdx
	leaq	.LC193(%rip), %rsi
	movl	$1, %edi
	movl	$1, %eax
	vdivsd	24(%rsp), %xmm1, %xmm0
	vmovsd	%xmm1, 32(%rsp)
	call	__printf_chk@PLT
	movl	$10, %edi
	call	putchar@PLT
	vxorpd	%xmm4, %xmm4, %xmm4
	vcvtsi2sdq	72(%rsp), %xmm4, %xmm0
	vmovsd	40(%rsp), %xmm2
	leaq	.LC194(%rip), %rsi
	movl	$1, %edi
	vmovsd	%xmm0, 24(%rsp)
	vdivsd	%xmm2, %xmm0, %xmm0
	movl	$1, %eax
	call	__printf_chk@PLT
	vmovsd	24(%rsp), %xmm5
	leaq	.LC195(%rip), %rsi
	movl	$1, %edi
	movl	$1, %eax
	vdivsd	16(%rsp), %xmm5, %xmm0
	call	__printf_chk@PLT
	vmovsd	24(%rsp), %xmm5
	vmovsd	32(%rsp), %xmm1
	leaq	.LC196(%rip), %rsi
	vdivsd	%xmm1, %xmm5, %xmm0
	movl	$1, %edi
	movl	$1, %eax
	call	__printf_chk@PLT
.L1983:
	cmpq	$1, 8(%rsp)
	jne	.L1981
	movq	%r14, %rdi
	call	nrand48@PLT
	cqto
	idivq	%rbx
	vxorpd	%xmm7, %xmm7, %xmm7
	leaq	.LC200(%rip), %rsi
	movl	$1, %edi
	movl	$1, %eax
	vcvtss2sd	0(%r13,%rdx,4), %xmm7, %xmm0
	call	__printf_chk@PLT
.L1981:
	movq	328(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L2016
	addq	$344, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	xorl	%eax, %eax
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.L2014:
	.cfi_restore_state
	leaq	.LC182(%rip), %rdi
	call	perror@PLT
.L1977:
	movq	stderr(%rip), %rdi
	movl	$100, %r8d
	leaq	.LC164(%rip), %rcx
	leaq	.LC184(%rip), %rdx
	movl	$1, %esi
	xorl	%eax, %eax
	call	__fprintf_chk@PLT
	orl	$-1, %r8d
	jmp	.L1986
.L1944:
	movslq	208(%rsp), %rax
	leaq	_ZL12long_options(%rip), %rdx
	salq	$5, %rax
	movq	(%rdx,%rax), %r9
	movq	stderr(%rip), %rdi
	movl	$606, %r8d
	leaq	.LC1(%rip), %rcx
	leaq	.LC174(%rip), %rdx
	movl	$1, %esi
	xorl	%eax, %eax
	call	__fprintf_chk@PLT
	jmp	.L1955
.L2015:
	leaq	.LC185(%rip), %rdi
	call	getenv@PLT
	leaq	.LC186(%rip), %rdi
	movq	%rax, 16(%rsp)
	call	getenv@PLT
	movq	16(%rsp), %r8
	movq	%rax, %r15
	testq	%r8, %r8
	je	.L1979
	testq	%rax, %rax
	jne	.L2017
.L1979:
	movq	%rbp, %rdx
	leaq	.LC188(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movq	%r12, %rdx
	leaq	.LC189(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	vxorpd	%xmm5, %xmm5, %xmm5
	vcvtsi2sdq	%r12, %xmm5, %xmm0
	vmovsd	%xmm0, 16(%rsp)
.L1980:
	leaq	.LC197(%rip), %rdi
	call	puts@PLT
	movl	$10, %edi
	call	putchar@PLT
	vxorpd	%xmm6, %xmm6, %xmm6
	vcvtsi2sdq	24(%rsp), %xmm6, %xmm1
	vcvtsi2sdq	%rbp, %xmm6, %xmm2
	movq	%rbx, %rdx
	leaq	.LC191(%rip), %rsi
	movl	$1, %edi
	vdivsd	%xmm1, %xmm2, %xmm0
	movl	$1, %eax
	vmovsd	%xmm2, 24(%rsp)
	vmovsd	%xmm1, 32(%rsp)
	call	__printf_chk@PLT
	vmovsd	16(%rsp), %xmm4
	vmovsd	32(%rsp), %xmm1
	movq	%rbx, %rdx
	vdivsd	%xmm1, %xmm4, %xmm0
	leaq	.LC192(%rip), %rsi
	movl	$1, %edi
	movl	$1, %eax
	call	__printf_chk@PLT
	movq	%rbx, %rdx
	leaq	.LC198(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movl	$10, %edi
	call	putchar@PLT
	vxorpd	%xmm6, %xmm6, %xmm6
	vcvtsi2sdq	72(%rsp), %xmm6, %xmm1
	vmovsd	24(%rsp), %xmm2
	leaq	.LC194(%rip), %rsi
	movl	$1, %edi
	vdivsd	%xmm2, %xmm1, %xmm0
	movl	$1, %eax
	vmovsd	%xmm1, 24(%rsp)
	call	__printf_chk@PLT
	vmovsd	24(%rsp), %xmm1
	leaq	.LC195(%rip), %rsi
	movl	$1, %edi
	movl	$1, %eax
	vdivsd	16(%rsp), %xmm1, %xmm0
	call	__printf_chk@PLT
	leaq	.LC199(%rip), %rdi
	call	puts@PLT
	jmp	.L1983
.L2017:
	movq	%r8, %rdi
	xorl	%esi, %esi
	call	strtod@PLT
	xorl	%esi, %esi
	movq	%r15, %rdi
	vmovsd	%xmm0, 32(%rsp)
	call	strtod@PLT
	vmovapd	%xmm0, %xmm1
	movq	stderr(%rip), %rdi
	vmovsd	%xmm0, 40(%rsp)
	vmovsd	32(%rsp), %xmm0
	leaq	.LC187(%rip), %rdx
	movl	$1, %esi
	movl	$2, %eax
	call	__fprintf_chk@PLT
	movq	stderr(%rip), %rdi
	call	fflush@PLT
	vxorpd	%xmm6, %xmm6, %xmm6
	vcvtsi2sdq	%r12, %xmm6, %xmm0
	vmovsd	40(%rsp), %xmm1
	movq	%rbp, %rdx
	leaq	.LC188(%rip), %rsi
	vmovsd	%xmm0, 16(%rsp)
	vmulsd	32(%rsp), %xmm0, %xmm0
	movl	$1, %edi
	xorl	%eax, %eax
	vdivsd	%xmm1, %xmm0, %xmm0
	vcvttsd2siq	%xmm0, %r15
	call	__printf_chk@PLT
	xorl	%eax, %eax
	movq	%r12, %rdx
	leaq	.LC189(%rip), %rsi
	movl	$1, %edi
	call	__printf_chk@PLT
	testq	%r15, %r15
	je	.L1980
	jmp	.L1984
.L1985:
	leaq	.LC183(%rip), %rdi
	movq	%rcx, 80(%rsp)
	call	perror@PLT
	movq	80(%rsp), %rcx
	movl	%ecx, %edi
	call	close@PLT
	jmp	.L1977
.L1943:
	movq	32(%rsp), %rdi
	call	_Z10parse_algoPKc
	movl	%eax, %r15d
	cmpl	$9, %eax
	je	.L2018
	cmpl	$8, %eax
	ja	.L1962
	leaq	.L1961(%rip), %rcx
	movl	%eax, %edx
	movslq	(%rcx,%rdx,4), %rax
	addq	%rcx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L1961:
	.long	.L1992-.L1961
	.long	.L1993-.L1961
	.long	.L1965-.L1961
	.long	.L1964-.L1961
	.long	.L1963-.L1961
	.long	.L1963-.L1961
	.long	.L1962-.L1961
	.long	.L1994-.L1961
	.long	.L1960-.L1961
	.section	.text.startup
.L1994:
	movq	%rbp, %rbx
	movl	$1, %r12d
.L1962:
	cmpq	$1, %rbx
	jle	.L1966
	movq	$1, 8(%rsp)
	xorl	%r8d, %r8d
	jmp	.L1957
.L1992:
	movl	$1, %r12d
	movl	$1, %ebx
	movl	$1, %ebp
.L1966:
	movq	%rbx, 8(%rsp)
	xorl	%r8d, %r8d
	jmp	.L1957
.L1993:
	movq	$1, 8(%rsp)
	xorl	%r8d, %r8d
	movl	$16, %ebx
	movl	$1, %r12d
	movl	$1, %ebp
	jmp	.L1957
.L2018:
	movq	$1, 8(%rsp)
	movl	$1, %r8d
	jmp	.L1957
.L1960:
	movq	%r12, %rcx
	imulq	%rbp, %rcx
	testq	%rbx, %rbx
	jne	.L1969
	movl	$1, %ebx
.L1969:
	leaq	-1(%rcx,%rbx), %rbx
	movq	%rbx, %rax
	cqto
	idivq	%rcx
	subq	%rdx, %rbx
	jmp	.L1962
.L1963:
	movq	%r12, %rcx
	salq	$4, %rcx
	testq	%rbx, %rbx
	jne	.L1968
	movl	$1, %ebx
.L1968:
	leaq	-1(%rcx,%rbx), %rbx
	movq	%rbx, %rax
	cqto
	idivq	%rcx
	movl	$1, %ebp
	subq	%rdx, %rbx
	jmp	.L1962
.L1964:
	testq	%rbx, %rbx
	jne	.L1967
	movl	$1, %ebx
.L1967:
	addq	$15, %rbx
	movq	%rbx, %rax
	movl	$16, %ecx
	cqto
	idivq	%rcx
	movl	$1, %r12d
	movl	$1, %ebp
	subq	%rdx, %rbx
	jmp	.L1962
.L1965:
	movq	%r12, %rbx
	salq	$4, %rbx
	movl	$1, %ebp
	jmp	.L1962
.L2012:
	movq	(%r14), %rdi
	movl	%r8d, 8(%rsp)
	call	_ZL5usagePKc
	movl	8(%rsp), %r8d
	movl	%r8d, %edi
	call	exit@PLT
.L2013:
	leaq	.LC181(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	err@PLT
.L2016:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE5666:
	.size	main, .-main
	.p2align 4
	.type	_GLOBAL__sub_I__Z11axpb_scalar14axpb_options_tfPff, @function
_GLOBAL__sub_I__Z11axpb_scalar14axpb_options_tfPff:
.LFB5864:
	.cfi_startproc
	endbr64
	leaq	_Z24axpb_simd_parallel_m_mnmILi1EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 48+axpb_funs_table(%rip)
	movq	%rdi, 120+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi2EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 160+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi2EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 184+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi2EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 192+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi3EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 232+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi3EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 256+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi3EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 264+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi4EEl14axpb_options_tfPff(%rip), %rdi
	leaq	_Z9axpb_simd14axpb_options_tfPff(%rip), %rcx
	leaq	_Z15axpb_simd_m_nmn14axpb_options_tfPff(%rip), %rax
	leaq	_Z11axpb_scalar14axpb_options_tfPff(%rip), %rsi
	leaq	_Z11axpb_simd_m14axpb_options_tfPff(%rip), %rdx
	movq	%rdi, 304+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi1EEl14axpb_options_tfPff(%rip), %r9
	leaq	_Z15axpb_simd_m_mnmILi1EEl14axpb_options_tfPff(%rip), %r8
	leaq	_Z15axpb_simd_m_mnmILi4EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rcx, 8+axpb_funs_table(%rip)
	movq	%rax, 32+axpb_funs_table(%rip)
	movq	%rcx, 80+axpb_funs_table(%rip)
	movq	%rax, 104+axpb_funs_table(%rip)
	movq	%rcx, 152+axpb_funs_table(%rip)
	movq	%rax, 176+axpb_funs_table(%rip)
	movq	%rcx, 224+axpb_funs_table(%rip)
	movq	%rax, 248+axpb_funs_table(%rip)
	movq	%rcx, 296+axpb_funs_table(%rip)
	movq	%rsi, axpb_funs_table(%rip)
	movq	%r9, 16+axpb_funs_table(%rip)
	movq	%rdx, 24+axpb_funs_table(%rip)
	movq	%r8, 40+axpb_funs_table(%rip)
	movq	%rsi, 72+axpb_funs_table(%rip)
	movq	%r9, 88+axpb_funs_table(%rip)
	movq	%rdx, 96+axpb_funs_table(%rip)
	movq	%r8, 112+axpb_funs_table(%rip)
	movq	%rsi, 144+axpb_funs_table(%rip)
	movq	%rdx, 168+axpb_funs_table(%rip)
	movq	%rsi, 216+axpb_funs_table(%rip)
	movq	%rdx, 240+axpb_funs_table(%rip)
	movq	%rsi, 288+axpb_funs_table(%rip)
	movq	%rdx, 312+axpb_funs_table(%rip)
	movq	%rax, 320+axpb_funs_table(%rip)
	movq	%rdi, 328+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi4EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 336+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi5EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 376+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi5EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 400+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi5EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 408+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi6EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 448+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi6EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 472+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi6EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 480+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi7EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 520+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi7EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 544+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi7EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 552+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi8EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 592+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi8EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 616+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi8EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 624+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi9EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rcx, 368+axpb_funs_table(%rip)
	movq	%rax, 392+axpb_funs_table(%rip)
	movq	%rcx, 440+axpb_funs_table(%rip)
	movq	%rax, 464+axpb_funs_table(%rip)
	movq	%rcx, 512+axpb_funs_table(%rip)
	movq	%rax, 536+axpb_funs_table(%rip)
	movq	%rcx, 584+axpb_funs_table(%rip)
	movq	%rax, 608+axpb_funs_table(%rip)
	movq	%rcx, 656+axpb_funs_table(%rip)
	movq	%rsi, 360+axpb_funs_table(%rip)
	movq	%rdx, 384+axpb_funs_table(%rip)
	movq	%rsi, 432+axpb_funs_table(%rip)
	movq	%rdx, 456+axpb_funs_table(%rip)
	movq	%rsi, 504+axpb_funs_table(%rip)
	movq	%rdx, 528+axpb_funs_table(%rip)
	movq	%rsi, 576+axpb_funs_table(%rip)
	movq	%rdx, 600+axpb_funs_table(%rip)
	movq	%rsi, 648+axpb_funs_table(%rip)
	movq	%rdi, 664+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi9EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 688+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi9EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 696+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi10EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 736+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi10EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 760+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi10EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 768+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi11EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 808+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi11EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 832+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi11EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 840+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi12EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 880+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi12EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 904+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi12EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 912+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi13EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 952+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi13EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 976+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi13EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 984+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi14EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rax, 680+axpb_funs_table(%rip)
	movq	%rcx, 728+axpb_funs_table(%rip)
	movq	%rax, 752+axpb_funs_table(%rip)
	movq	%rcx, 800+axpb_funs_table(%rip)
	movq	%rax, 824+axpb_funs_table(%rip)
	movq	%rcx, 872+axpb_funs_table(%rip)
	movq	%rax, 896+axpb_funs_table(%rip)
	movq	%rcx, 944+axpb_funs_table(%rip)
	movq	%rax, 968+axpb_funs_table(%rip)
	movq	%rdx, 672+axpb_funs_table(%rip)
	movq	%rsi, 720+axpb_funs_table(%rip)
	movq	%rdx, 744+axpb_funs_table(%rip)
	movq	%rsi, 792+axpb_funs_table(%rip)
	movq	%rdx, 816+axpb_funs_table(%rip)
	movq	%rsi, 864+axpb_funs_table(%rip)
	movq	%rdx, 888+axpb_funs_table(%rip)
	movq	%rsi, 936+axpb_funs_table(%rip)
	movq	%rdx, 960+axpb_funs_table(%rip)
	movq	%rsi, 1008+axpb_funs_table(%rip)
	movq	%rdi, 1024+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi14EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1048+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi14EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1056+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi15EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1096+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi15EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1120+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi15EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1128+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi16EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1168+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi16EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1192+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi16EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1200+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi17EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1240+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi17EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1264+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi17EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1272+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi18EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1312+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi18EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rcx, 1016+axpb_funs_table(%rip)
	movq	%rax, 1040+axpb_funs_table(%rip)
	movq	%rcx, 1088+axpb_funs_table(%rip)
	movq	%rax, 1112+axpb_funs_table(%rip)
	movq	%rcx, 1160+axpb_funs_table(%rip)
	movq	%rax, 1184+axpb_funs_table(%rip)
	movq	%rcx, 1232+axpb_funs_table(%rip)
	movq	%rax, 1256+axpb_funs_table(%rip)
	movq	%rcx, 1304+axpb_funs_table(%rip)
	movq	%rax, 1328+axpb_funs_table(%rip)
	movq	%rdx, 1032+axpb_funs_table(%rip)
	movq	%rsi, 1080+axpb_funs_table(%rip)
	movq	%rdx, 1104+axpb_funs_table(%rip)
	movq	%rsi, 1152+axpb_funs_table(%rip)
	movq	%rdx, 1176+axpb_funs_table(%rip)
	movq	%rsi, 1224+axpb_funs_table(%rip)
	movq	%rdx, 1248+axpb_funs_table(%rip)
	movq	%rsi, 1296+axpb_funs_table(%rip)
	movq	%rdx, 1320+axpb_funs_table(%rip)
	movq	%rdi, 1336+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi18EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1344+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi19EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1384+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi19EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1408+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi19EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1416+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi20EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1456+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi20EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1480+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi20EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1488+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi21EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1528+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi21EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1552+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi21EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1560+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi22EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1600+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi22EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1624+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi22EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1632+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi23EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1672+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi23EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rcx, 1376+axpb_funs_table(%rip)
	movq	%rax, 1400+axpb_funs_table(%rip)
	movq	%rcx, 1448+axpb_funs_table(%rip)
	movq	%rax, 1472+axpb_funs_table(%rip)
	movq	%rcx, 1520+axpb_funs_table(%rip)
	movq	%rax, 1544+axpb_funs_table(%rip)
	movq	%rcx, 1592+axpb_funs_table(%rip)
	movq	%rax, 1616+axpb_funs_table(%rip)
	movq	%rcx, 1664+axpb_funs_table(%rip)
	movq	%rsi, 1368+axpb_funs_table(%rip)
	movq	%rdx, 1392+axpb_funs_table(%rip)
	movq	%rsi, 1440+axpb_funs_table(%rip)
	movq	%rdx, 1464+axpb_funs_table(%rip)
	movq	%rsi, 1512+axpb_funs_table(%rip)
	movq	%rdx, 1536+axpb_funs_table(%rip)
	movq	%rsi, 1584+axpb_funs_table(%rip)
	movq	%rdx, 1608+axpb_funs_table(%rip)
	movq	%rsi, 1656+axpb_funs_table(%rip)
	movq	%rdx, 1680+axpb_funs_table(%rip)
	movq	%rdi, 1696+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi23EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1704+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi24EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1744+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi24EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1768+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi24EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1776+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi25EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1816+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi25EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1840+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi25EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1848+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi26EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1888+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi26EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1912+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi26EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1920+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi27EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1960+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi27EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1984+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi27EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 1992+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi28EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rax, 1688+axpb_funs_table(%rip)
	movq	%rcx, 1736+axpb_funs_table(%rip)
	movq	%rax, 1760+axpb_funs_table(%rip)
	movq	%rcx, 1808+axpb_funs_table(%rip)
	movq	%rax, 1832+axpb_funs_table(%rip)
	movq	%rcx, 1880+axpb_funs_table(%rip)
	movq	%rax, 1904+axpb_funs_table(%rip)
	movq	%rcx, 1952+axpb_funs_table(%rip)
	movq	%rax, 1976+axpb_funs_table(%rip)
	movq	%rsi, 1728+axpb_funs_table(%rip)
	movq	%rdx, 1752+axpb_funs_table(%rip)
	movq	%rsi, 1800+axpb_funs_table(%rip)
	movq	%rdx, 1824+axpb_funs_table(%rip)
	movq	%rsi, 1872+axpb_funs_table(%rip)
	movq	%rdx, 1896+axpb_funs_table(%rip)
	movq	%rsi, 1944+axpb_funs_table(%rip)
	movq	%rdx, 1968+axpb_funs_table(%rip)
	movq	%rsi, 2016+axpb_funs_table(%rip)
	movq	%rcx, 2024+axpb_funs_table(%rip)
	movq	%rdi, 2032+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi28EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2056+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi28EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2064+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi29EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2104+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi29EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2128+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi29EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2136+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi30EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2176+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi30EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2200+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi30EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2208+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi31EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2248+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi31EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2272+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi31EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2280+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi32EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2320+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi32EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2344+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi32EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rax, 2048+axpb_funs_table(%rip)
	movq	%rcx, 2096+axpb_funs_table(%rip)
	movq	%rax, 2120+axpb_funs_table(%rip)
	movq	%rcx, 2168+axpb_funs_table(%rip)
	movq	%rax, 2192+axpb_funs_table(%rip)
	movq	%rcx, 2240+axpb_funs_table(%rip)
	movq	%rax, 2264+axpb_funs_table(%rip)
	movq	%rcx, 2312+axpb_funs_table(%rip)
	movq	%rax, 2336+axpb_funs_table(%rip)
	movq	%rdx, 2040+axpb_funs_table(%rip)
	movq	%rsi, 2088+axpb_funs_table(%rip)
	movq	%rdx, 2112+axpb_funs_table(%rip)
	movq	%rsi, 2160+axpb_funs_table(%rip)
	movq	%rdx, 2184+axpb_funs_table(%rip)
	movq	%rsi, 2232+axpb_funs_table(%rip)
	movq	%rdx, 2256+axpb_funs_table(%rip)
	movq	%rsi, 2304+axpb_funs_table(%rip)
	movq	%rdx, 2328+axpb_funs_table(%rip)
	movq	%rdi, 2352+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi33EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2392+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi33EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2416+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi33EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2424+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi34EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2464+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi34EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2488+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi34EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2496+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi35EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2536+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi35EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2560+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi35EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2568+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi36EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2608+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi36EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2632+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi36EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2640+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi37EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2680+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi37EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rcx, 2384+axpb_funs_table(%rip)
	movq	%rax, 2408+axpb_funs_table(%rip)
	movq	%rcx, 2456+axpb_funs_table(%rip)
	movq	%rax, 2480+axpb_funs_table(%rip)
	movq	%rcx, 2528+axpb_funs_table(%rip)
	movq	%rax, 2552+axpb_funs_table(%rip)
	movq	%rcx, 2600+axpb_funs_table(%rip)
	movq	%rax, 2624+axpb_funs_table(%rip)
	movq	%rcx, 2672+axpb_funs_table(%rip)
	movq	%rsi, 2376+axpb_funs_table(%rip)
	movq	%rdx, 2400+axpb_funs_table(%rip)
	movq	%rsi, 2448+axpb_funs_table(%rip)
	movq	%rdx, 2472+axpb_funs_table(%rip)
	movq	%rsi, 2520+axpb_funs_table(%rip)
	movq	%rdx, 2544+axpb_funs_table(%rip)
	movq	%rsi, 2592+axpb_funs_table(%rip)
	movq	%rdx, 2616+axpb_funs_table(%rip)
	movq	%rsi, 2664+axpb_funs_table(%rip)
	movq	%rdx, 2688+axpb_funs_table(%rip)
	movq	%rax, 2696+axpb_funs_table(%rip)
	movq	%rdi, 2704+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi37EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2712+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi38EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2752+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi38EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2776+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi38EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2784+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi39EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2824+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi39EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2848+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi39EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2856+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi40EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2896+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi40EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2920+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi40EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2928+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi41EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2968+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi41EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 2992+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi41EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 3000+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi42EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rcx, 2744+axpb_funs_table(%rip)
	movq	%rax, 2768+axpb_funs_table(%rip)
	movq	%rcx, 2816+axpb_funs_table(%rip)
	movq	%rax, 2840+axpb_funs_table(%rip)
	movq	%rcx, 2888+axpb_funs_table(%rip)
	movq	%rax, 2912+axpb_funs_table(%rip)
	movq	%rcx, 2960+axpb_funs_table(%rip)
	movq	%rax, 2984+axpb_funs_table(%rip)
	movq	%rcx, 3032+axpb_funs_table(%rip)
	movq	%rsi, 2736+axpb_funs_table(%rip)
	movq	%rdx, 2760+axpb_funs_table(%rip)
	movq	%rsi, 2808+axpb_funs_table(%rip)
	movq	%rdx, 2832+axpb_funs_table(%rip)
	movq	%rsi, 2880+axpb_funs_table(%rip)
	movq	%rdx, 2904+axpb_funs_table(%rip)
	movq	%rsi, 2952+axpb_funs_table(%rip)
	movq	%rdx, 2976+axpb_funs_table(%rip)
	movq	%rsi, 3024+axpb_funs_table(%rip)
	movq	%rdi, 3040+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi42EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 3064+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi42EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 3072+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi43EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 3112+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi43EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 3136+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi43EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 3144+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi44EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 3184+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi44EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 3208+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi44EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 3216+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi45EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 3256+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi45EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 3280+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi45EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 3288+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi46EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 3328+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi46EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 3352+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi46EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 3360+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi47EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rax, 3056+axpb_funs_table(%rip)
	movq	%rcx, 3104+axpb_funs_table(%rip)
	movq	%rax, 3128+axpb_funs_table(%rip)
	movq	%rcx, 3176+axpb_funs_table(%rip)
	movq	%rax, 3200+axpb_funs_table(%rip)
	movq	%rcx, 3248+axpb_funs_table(%rip)
	movq	%rax, 3272+axpb_funs_table(%rip)
	movq	%rcx, 3320+axpb_funs_table(%rip)
	movq	%rax, 3344+axpb_funs_table(%rip)
	movq	%rdx, 3048+axpb_funs_table(%rip)
	movq	%rsi, 3096+axpb_funs_table(%rip)
	movq	%rdx, 3120+axpb_funs_table(%rip)
	movq	%rsi, 3168+axpb_funs_table(%rip)
	movq	%rdx, 3192+axpb_funs_table(%rip)
	movq	%rsi, 3240+axpb_funs_table(%rip)
	movq	%rdx, 3264+axpb_funs_table(%rip)
	movq	%rsi, 3312+axpb_funs_table(%rip)
	movq	%rdx, 3336+axpb_funs_table(%rip)
	movq	%rsi, 3384+axpb_funs_table(%rip)
	movq	%rdi, 3400+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi47EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 3424+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi47EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rdi, 3432+axpb_funs_table(%rip)
	leaq	_Z11axpb_simd_cILi48EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rax, 3416+axpb_funs_table(%rip)
	movq	%rdi, 3472+axpb_funs_table(%rip)
	movq	%rax, 3488+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi48EEl14axpb_options_tfPff(%rip), %rdi
	movq	%rax, 3560+axpb_funs_table(%rip)
	leaq	_Z15axpb_simd_m_mnmILi49EEl14axpb_options_tfPff(%rip), %rax
	movq	%rcx, 3392+axpb_funs_table(%rip)
	movq	%rcx, 3464+axpb_funs_table(%rip)
	movq	%rdi, 3496+axpb_funs_table(%rip)
	movq	%rcx, 3536+axpb_funs_table(%rip)
	movq	%rax, 3568+axpb_funs_table(%rip)
	leaq	_Z24axpb_simd_parallel_m_mnmILi48EEl14axpb_options_tfPff(%rip), %rdi
	leaq	_Z11axpb_simd_cILi49EEl14axpb_options_tfPff(%rip), %rcx
	leaq	_Z24axpb_simd_parallel_m_mnmILi49EEl14axpb_options_tfPff(%rip), %rax
	movq	%rdx, 3408+axpb_funs_table(%rip)
	movq	%rsi, 3456+axpb_funs_table(%rip)
	movq	%rdx, 3480+axpb_funs_table(%rip)
	movq	%rdi, 3504+axpb_funs_table(%rip)
	movq	%rsi, 3528+axpb_funs_table(%rip)
	movq	%rcx, 3544+axpb_funs_table(%rip)
	movq	%rdx, 3552+axpb_funs_table(%rip)
	movq	%rax, 3576+axpb_funs_table(%rip)
	ret
	.cfi_endproc
.LFE5864:
	.size	_GLOBAL__sub_I__Z11axpb_scalar14axpb_options_tfPff, .-_GLOBAL__sub_I__Z11axpb_scalar14axpb_options_tfPff
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I__Z11axpb_scalar14axpb_options_tfPff
	.section	.rodata.str1.1
.LC201:
	.string	"algo"
.LC202:
	.string	"cuda-block-size"
.LC203:
	.string	"active-threads-per-warp"
.LC204:
	.string	"concurrent-vars"
.LC205:
	.string	"vars"
.LC206:
	.string	"n"
.LC207:
	.string	"seed"
.LC208:
	.string	"help"
	.section	.data.rel.local,"aw"
	.align 32
	.type	_ZL12long_options, @object
	.size	_ZL12long_options, 288
_ZL12long_options:
	.quad	.LC201
	.long	1
	.zero	4
	.quad	0
	.long	97
	.zero	4
	.quad	.LC202
	.long	1
	.zero	4
	.quad	0
	.long	98
	.zero	4
	.quad	.LC203
	.long	1
	.zero	4
	.quad	0
	.long	119
	.zero	4
	.quad	.LC204
	.long	1
	.zero	4
	.quad	0
	.long	99
	.zero	4
	.quad	.LC205
	.long	1
	.zero	4
	.quad	0
	.long	109
	.zero	4
	.quad	.LC206
	.long	1
	.zero	4
	.quad	0
	.long	110
	.zero	4
	.quad	.LC207
	.long	1
	.zero	4
	.quad	0
	.long	115
	.zero	4
	.quad	.LC208
	.long	0
	.zero	4
	.quad	0
	.long	104
	.zero	4
	.quad	0
	.long	0
	.zero	4
	.quad	0
	.long	0
	.zero	4
	.section	.rodata.str1.1
.LC209:
	.string	"simd"
.LC210:
	.string	"simd_c"
.LC211:
	.string	"simd_m"
.LC212:
	.string	"simd_m_nmn"
.LC213:
	.string	"simd_m_mnm"
.LC214:
	.string	"simd_parallel_m_mnm"
.LC215:
	.string	"cuda"
.LC216:
	.string	"cuda_c"
	.section	.data.rel.local
	.align 32
	.type	_ZL10algo_table, @object
	.size	_ZL10algo_table, 144
_ZL10algo_table:
	.long	0
	.zero	4
	.quad	.LC152
	.long	1
	.zero	4
	.quad	.LC209
	.long	2
	.zero	4
	.quad	.LC210
	.long	3
	.zero	4
	.quad	.LC211
	.long	4
	.zero	4
	.quad	.LC212
	.long	5
	.zero	4
	.quad	.LC213
	.long	6
	.zero	4
	.quad	.LC214
	.long	7
	.zero	4
	.quad	.LC215
	.long	8
	.zero	4
	.quad	.LC216
	.globl	axpb_funs_table
	.bss
	.align 32
	.type	axpb_funs_table, @object
	.size	axpb_funs_table, 3600
axpb_funs_table:
	.zero	3600
	.ident	"GCC: (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	 1f - 0f
	.long	 4f - 1f
	.long	 5
0:
	.string	 "GNU"
1:
	.align 8
	.long	 0xc0000002
	.long	 3f - 2f
2:
	.long	 0x3
3:
	.align 8
4:
