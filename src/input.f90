SUBROUTINE read_input_file(FILENAME)
   use comin
   use comcout
   implicit none
   integer :: I, ioerr, LU = 16
   CHARACTER*(*) FILENAME
   character(len=200) :: line
   character(len=1) :: first_char

   if (FILENAME .EQ. '') then
      write (*, *) 'ERROR model filename not given!'
      stop
   else
      if (printDebugInfo) write (*, *) '# read_input_file: file = ', filename
      OPEN (UNIT=LU, FILE=FILENAME, iostat=ioerr)
      if (ioerr /= 0) then ! error opening file
         WRITE (*, *) 'READ_INPUT_FILE: ERROR IN OPEN, FILE = ', FILENAME
         RETURN
      end if

      ! Skip comment lines and read data
      call skip_comments_and_read(LU, DN)
      call skip_comments_and_read(LU, TA)
      call skip_comments_and_read(LU, DD)
      call skip_comments_and_read(LU, AA(:,1))
      call skip_comments_and_read(LU, AA(:,2))
      call skip_comments_and_read(LU, ASS)
      call skip_comments_and_read(LU, VD)
      call skip_comments_and_read_single(LU, XC)
      call skip_comments_and_read_single(LU, PZL)

      CLOSE (LU)
      RETURN
   end if
999 WRITE (*, *) 'READ_INPUT_FILE: ERROR IN READ, FILE = ', FILENAME
   CLOSE (LU)
   RETURN

contains
   subroutine skip_comments_and_read_single(unit, var)
      integer, intent(in) :: unit
      real(kind=8), intent(out) :: var
      character(len=200) :: line
      
      do
         read(unit, '(A)', iostat=ioerr) line
         if (ioerr /= 0) return
         line = adjustl(line)
         if (len_trim(line) == 0) cycle  ! skip empty lines
         if (line(1:1) == '#' .or. line(1:1) == '!') cycle  ! skip comments
         read(line, *, iostat=ioerr) var
         if (ioerr == 0) exit
      end do
   end subroutine

   subroutine skip_comments_and_read(unit, array)
      integer, intent(in) :: unit
      real(kind=8), intent(out) :: array(:)
      character(len=200) :: line
      
      do
         read(unit, '(A)', iostat=ioerr) line
         if (ioerr /= 0) return
         line = adjustl(line)
         if (len_trim(line) == 0) cycle  ! skip empty lines
         if (line(1:1) == '#' .or. line(1:1) == '!') cycle  ! skip comments
         read(line, *, iostat=ioerr) array
         if (ioerr == 0) exit
      end do
   end subroutine
END