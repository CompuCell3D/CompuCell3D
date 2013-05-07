unit uRRList;

{ Copyright 2012 Herbert M Sauro

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   In plain english this means:

   You CAN freely download and use this software, in whole or in part, for personal,
   company internal, or commercial purposes.

   You CAN use the software in packages or distributions that you create.

   You SHOULD include a copy of the license in any redistribution you may make.

   You are NOT required include the source of software, or of any modifications you may
   have made to it, in any redistribution you may assemble that includes it.

   YOU CANNOT:

   redistribute any piece of this software without proper attribution;
}


interface

Uses SysUtils, Classes;
//, uSBWCommon, uSBWComplex, uSBWArray;

type
  TRRList = class;

  TRRListType = (rrList, rrInteger, rrDouble, rrString);

  TRRListItem = class (TObject)
            public
                DataType : TRRListType;
                iValue : integer;
                dValue : double;
                sValue : AnsiString;
                list : TRRList;
                function getInteger : Double;
                function getDouble : double;
                function getString : AnsiString;
                function getList : TRRList;

                constructor Create (iValue : integer); overload;
                constructor Create (dValue : double);  overload;
                constructor Create (sValue : AnsiString); overload;
                constructor Create (list : TRRList); overload;

                destructor  Destroy; override;
  end;


  TRRList = class (TList)
             protected
               function Get (Index : integer) : TRRListItem;
               procedure Put (Index : integer; Item : TRRListItem);
             public
               destructor Destroy; override;
               function  Add (Item : TRRListItem) : integer;
               procedure Delete (Index : Integer);
               procedure copy (src : TRRList);
               property  Items[Index : integer] : TRRListItem read Get write Put; default;
               //function  Count : integer;
  end;

implementation


constructor TRRListItem.Create (iValue : integer);
begin
  inherited Create;
  DataType := rrInteger;
  Self.iValue := iValue;
end;


constructor TRRListItem.Create (dValue : double);
begin
  inherited Create;
  DataType := rrDouble;
  Self.dValue := dValue;
end;


constructor TRRListItem.Create (sValue : AnsiString);
begin
  inherited Create;
  DataType := rrString;
  Self.sValue := sValue;
end;


constructor TRRListItem.Create (list : TRRList);
begin
  inherited Create;
  DataType := rrList;
  Self.list := List;
end;


destructor TRRListItem.Destroy;
begin
  case DataType of
     rrList : List.Free;
  end;
  inherited Destroy;
end;


// --------------------------------------------------------------------------

function TRRListItem.getInteger : Double;
begin
  if DataType <> rrInteger then
     raise Exception.Create ('Integer expected in List item');
  result := iValue;
end;


function TRRListItem.getDouble : Double;
begin
  if DataType <> rrDouble then
     raise Exception.Create ('Double expected in List item');
  result := dValue;
end;


function TRRListItem.getString : AnsiString;
begin
  if DataType <> rrString then
     raise Exception.Create ('String expected in List item');
  result := sValue;
end;


function TRRListItem.getList : TRRList;
begin
  if DataType <> rrList then
     raise Exception.Create ('List expected in List item');
  result := list;
end;

// ----------------------------------------------------------------------


function TRRList.Get (Index : integer) : TRRListItem;
begin
  result := TRRListItem(inherited Get(index));
end;


procedure TRRList.Put (Index : integer; Item : TRRListItem);
begin
  inherited Put (Index, Item);
end;

function TRRList.Add (Item : TRRListItem) : integer;
begin
  result := inherited Add (Item);
end;


procedure TRRList.Delete (Index : Integer);
begin
  Items[Index].Free;
  Items[Index] := nil;
  inherited Delete (Index);
end;


destructor TRRList.Destroy;
var i : integer;
begin
  for i := 0 to Count - 1 do
      Items[i].Free;
  inherited Destroy;
end;


procedure TRRList.copy (src: TRRList);
var i : integer;
begin
  self.Clear;
  for i := 0 to src.count - 1 do
      begin
      case src[i].DataType of
          rrInteger  : self.add (TRRListItem.Create(src[i].iValue));
          rrDouble  : self.add (TRRListItem.Create(src[i].dValue));
          rrString  : self.add (TRRListItem.Create(src[i].sValue));
          rrList    : self.add (TRRListItem.Create(src[i].list));
      else
          raise Exception.Create ('Unknown data type while copying List');
      end;
      end;
end;


// ----------------------------------------------------------------------


end.

